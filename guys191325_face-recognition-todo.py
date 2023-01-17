!pip install git+https://github.com/goolig/dsClass.git

from dsClass.path_helper import *
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

cwd = os.getcwd()



# change directory to the dataset where our

# custom scripts are found

#os.chdir("/kaggle/input/")

try:

    from dsClass.align_custom import AlignCustom

    from dsClass.face_feature import FaceFeature

    from dsClass.mtcnn_detect import MTCNNDetect

    from dsClass.tf_graph import FaceRecGraph

    from dsClass.path_helper import *

except ModuleNotFoundError:  # for non-kaggle environemnt

    from align_custom import AlignCustom

    from face_feature import FaceFeature

    from mtcnn_detect import MTCNNDetect

    from tf_graph import FaceRecGraph

    get_file_path = lambda x:cwd

    from helper import vid2vid_audio_transfer



import sys

import glob

import cv2

import json

import numpy as np

import pandas as pd

import time

import scipy

import urllib

import matplotlib.pyplot as plt
from IPython.display import YouTubeVideo

YouTubeVideo('NuhCoO6GO5U')
def annotate_face(rect, recog_data, frame):

    """

    Draw a box around the face and label the person

    :param rect : the face bounding box

    :param recog_data : tuple of person name and confidence percentage

    :param frame : the frame to draw on

    """

    shrtname = short_name(recog_data[0])

    acc = round(recog_data[1], 1)

    bbox_color = (255, 255, 255) if "Unknown" in recog_data[0] else (124,252,0)  # RGB



    #draw bounding box / fancy border for the face

    #cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),bbox_color) 

    draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), bbox_color, 1, 10, 10)  

    anot_text = shrtname + "-" + str(acc) + "%"

    cv2.putText(frame, anot_text,

                (rect[0]-4,rect[1]-4), cv2.FONT_HERSHEY_SIMPLEX,0.35,

                bbox_color, 1, cv2.LINE_AA)                        



    

def short_name(name):

    """e.g: Jaime Lannister -> Jaime.L"""

    if not name.startswith("Unknown"):

        name_split = name.split(" ")

        short_name = name_split[0]

        if len(name_split) > 1:

            short_name = short_name + "." + name_split[1][0]

        return(short_name)

    else:

        return name

    

    

def draw_border(img, pt1, pt2, color, thickness, r, d):

    """

    Fancy box drawing function by Dan Masek

    Code in: https://www.codemade.io/fast-and-accurate-face-tracking-in-live-video-with-python/

    """

    x1, y1 = pt1

    x2, y2 = pt2

 

    # Top left drawing

    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)

    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)

    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

 

    # Top right drawing

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)

    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)

    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

 

    # Bottom left drawing

    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)

    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)

    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

 

    # Bottom right drawing

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)

    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)

    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness) 

    

    

def read_image_from_url(url2read):

    req = urllib.request.urlopen(url2read)

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    img = cv2.imdecode(arr, -1) # 'Load it as it is'

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return(img)
dict_faces = dict()

dict_faces["Jaime Lannister"] = ["https://s2.r29static.com//bin/entry/97f/340x408,85/1832698/image.jpg",

                                "https://upload.wikimedia.org/wikipedia/en/thumb/b/b4/Jaime_Lannister-Nikolaj_Coster-Waldau.jpg/220px-Jaime_Lannister-Nikolaj_Coster-Waldau.jpg",

                                 "https://upload.wikimedia.org/wikipedia/pt/thumb/0/06/Nikolaj-Coster-Waldau-Game-of-Thrones.jpg/220px-Nikolaj-Coster-Waldau-Game-of-Thrones.jpg",

                                 "https://purewows3.imgix.net/images/articles/2017_09/jaime-lannister-season-7-game-of-thrones-finale1.jpg?auto=format,compress&cs=strip&fit=min&w=728&h=404",

                                 "https://cdn.newsday.com/polopoly_fs/1.13944684.1502107079!/httpImage/image.jpeg_gen/derivatives/landscape_768/image.jpeg",

                                 "https://www.cheatsheet.com/wp-content/uploads/2017/08/Jaime-Lannister-Game-of-Thrones.png",

                                 "https://fsmedia.imgix.net/9c/c0/27/10/15e0/44a4/8ecb/9339993b563d/nikolaj-coster-waldau-as-jaime-lannister-in-game-of-thrones-season-7.png?rect=0%2C0%2C1159%2C580&dpr=2&auto=format%2Ccompress&w=650",

                                 "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrIQuBKKUocAizwfWtIdhAcvfowLJatKqqDsO3ywYdh3rv-mBk"]

dict_faces['Cersei Lannister'] = ['https://cdn.pastemagazine.com/www/articles/CERSEI-LANNISTER-quotes-list.jpg',

                                 'https://assets3.thrillist.com/v1/image/2823203/size/gn-gift_guide_variable_c.jpg',

                                 'https://s3-us-west-2.amazonaws.com/flx-editorial-wordpress/wp-content/uploads/2017/07/13060545/Cersei-Lannister.jpg',

                                 'https://www.thewrap.com/wp-content/uploads/2018/07/cersei_lannister-1.jpg',

                                 'https://imagesvc.timeincapp.com/v3/fan/image?url=https%3A%2F%2Fwinteriscoming.net%2Ffiles%2F2018%2F09%2FCersei-Lannister.jpg&w=736&h=485&c=sc',

                                 'https://s1.r29static.com/bin/entry/b52/720x864,85/2180384/image.webp']



# Adding Cersei side profile

dict_faces['Cersei Lannister'].append('https://img1.looper.com/img/gallery/the-worst-things-cersei-lannister-has-ever-done/intro-1557760232.jpg')



dict_faces['Tyrion Lannister'] = ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4Re6vnUjEBxywRkRLrCmTMLf4YMWUF19GYS3cbKuRzVoTCz1d&s',

                                  'https://www.hbo.com/content/dam/hbodata/series/game-of-thrones/character/s5/tyrion-lannister-1920.jpg/_jcr_content/renditions/cq5dam.web.1200.675.jpeg',

                                  'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/44ccdd999a4b20cdaef4696b8bf1a2c502db2b66f6909ae2a8a96a01a698370688e9b73dad75f0d33ea170c58a251155-1-1558322295.jpg?crop=0.668xw:1.00xh;0.0918xw,0&resize=480:*',

                                  'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/theory-1553634761.jpg?crop=0.501xw:1.00xh;0,0&resize=480:*',

                                  'https://fromheartopaper.files.wordpress.com/2019/09/picsart_09-11-02.35.13.png?w=1080',

                                  'https://upload.wikimedia.org/wikipedia/en/thumb/5/50/Tyrion_Lannister-Peter_Dinklage.jpg/220px-Tyrion_Lannister-Peter_Dinklage.jpg']

dict_faces['Euron Greyjoy'] = ['http://assets.viewers-guide.hbo.com/small597a20afe7893@2x.jpg',

                                'http://static1.squarespace.com/static/52fc05c9e4b08fc45bd99090/5331dfe2e4b0f77da2ddde51/5cd9bdc8cdc8f30001b28a0c/1557848897813/cq5dam.web.1280.1280.jpeg?format=1500w',

                                'https://vignette.wikia.nocookie.net/gameofthrones/images/f/fc/Euron-Profile.PNG/revision/latest?cb=20170916200257',

                                'https://images.radiox.co.uk/images/52162?crop=16_9&width=660&relax=1&signature=k6BDCtKEIBn0Vy8EeqjdLe_JpaI=',

                                'https://cdn.vox-cdn.com/thumbor/iGnjL2BCwm_o_iyLu40Y00IjU_s=/0x0:2560x1394/1200x0/filters:focal(0x0:2560x1394):no_upscale()/cdn.vox-cdn.com/uploads/chorus_asset/file/16220189/Screen_Shot_2019_05_06_at_7.25.08_AM.png']

dict_faces['Qyburn'] = ['https://qph.fs.quoracdn.net/main-qimg-e2126ec618d9331d266d49eb09f2cf40.webp',

                        'https://qph.fs.quoracdn.net/main-qimg-53106ed4400757d745fef3da46dc02a1.webp',

                        'https://i.ytimg.com/vi/1Tsjj9o47n0/maxresdefault.jpg',

                        'https://i.redd.it/mm9sgp28ri811.jpg',

                        'https://vignette.wikia.nocookie.net/gameofthrones/images/2/2a/Qyburn_3x01.jpg/revision/latest/top-crop/width/300/height/300?cb=20130502025950',

                        'https://vignette.wikia.nocookie.net/gameofthrones/images/8/8a/804_Qyburn_Profile.png/revision/latest?cb=20190508024406']

dict_faces['Jon Snow'] = ['https://www.esquireme.com/sites/default/files/styles/full_img/public/images/2019/05/20/Game-of-Thrones-Season-8-Ep-6-Finale-%284%29.png?itok=1LN_Lkkk',

                        'https://images2.minutemediacdn.com/image/upload/c_crop,h_840,w_1500,x_0,y_6/f_auto,q_auto,w_1100/v1555003564/shape/mentalfloss/jon_snow_hed.jpg',

                        'https://d.newsweek.com/en/full/507446/jon-snow-game-thrones.jpg',

                        'https://www.indiewire.com/wp-content/uploads/2019/04/Helen-Sloan-HBO-4-copy.jpg?w=780',

                        'https://cdn.vox-cdn.com/thumbor/o2AXRjdoyonKroOEsxQjYWvtG-U=/99x0:1179x810/1200x800/filters:focal(99x0:1179x810)/cdn.vox-cdn.com/uploads/chorus_image/image/46094226/Jon_snow.0.jpg',

                        'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/jon-snow-rhaegar-targaryen-1554321615.jpg?crop=0.482xw:0.962xh;0,0&resize=480:*',

                        'https://image.insider.com/5cb3c8e96afbee373d4f2b62?width=1100&format=jpeg&auto=webp',

                        'https://vignette.wikia.nocookie.net/gameofthrones/images/d/d0/JonSnow8x06.PNG/revision/latest?cb=20190714094440']

dict_faces['Davos Seaworth'] = ['https://newscast183978364.files.wordpress.com/2019/06/davos-seaworth-1.jpg?w=400&h=280&crop=1',

                                'data:image/jpeg;base64,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',

                                'https://amp.thenational.ae/image/policy:1.823364:1557570524/lf08-Feb-GoTPics.jpg?f=16x9&w=1200&$p$f$w=b6fcdf1',

                                'https://vignette.wikia.nocookie.net/p__/images/b/b0/Davos_Seaworth_2.jpg/revision/latest?cb=20170125224137&path-prefix=protagonist',

                                'https://vignette.wikia.nocookie.net/gameofthrones/images/9/9d/DAVOSINFOBOXBELLS.PNG/revision/latest?cb=20190513052340']

dict_faces['Jorah Mormont'] = ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTa_dvQnskjXXOWJN5QWrEwG8YLihVOhJ8Kwa5r2nqxEMuBYMUY&s',

                                'data:image/jpeg;base64,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',

                                'https://ewedit.files.wordpress.com/2019/04/jorah.jpg',

                                'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMWFhUXGBgYFxcVGBcaGBcXGBodFxgdGBgeHSggGBolHRcYITEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OGhAQGi0lHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAQkAvgMBIgACEQEDEQH/xAAcAAACAwEBAQEAAAAAAAAAAAAEBQIDBgcBAAj/xABCEAABAwIEAwYDBgQEBQUBAAABAgMRACEEBRIxQVFhBhMicYGRMqGxB0JSwdHwIzNy4RRigvEVJENzslSSorPCNP/EABkBAAMBAQEAAAAAAAAAAAAAAAECAwAEBf/EACARAAIDAAIDAQEBAAAAAAAAAAABAhEhEjEDQVETYTL/2gAMAwEAAhEDEQA/ANviGiauw7UCiHE0IpcGi0iCDVNimeBTApbhF6jTdtNqSPZb0GJNQdVFeti1VYkVQmCvGqBUlk14k3oiFgFq8ivQalpomIJTX2mrAivdFMCysJqWmrUt1HFPoaSVOKAH725mg5JBUWz5KKkRS/8A4pqEpSpKbStUCxE2BEk9Im9QVnSLxrIG50pj3kRU35YjrwyGQr5xFC4fNWz8RKDE+MaR7/qaYNFKhYgjgQQZ9RTKaYPzkgAIvRaBaqn0RViDaiKVu0E5vReINDEUWA+bmrdE1NhPOrVRQCLHEWpLjVRWlLYNKMwwU0slaMinKsRtWoYVIrK4HBlJrS4U2qEbRVB6DXjwkVWk1NRtVkwMAeTVSBRTomqAmqEySE1ekVFsVmu2XaxrCAIKj3itkp36kmbAc+cdYVukGMeTHGZ5s2zAUfEdhefl/tQAzlJBUVKKTbSkBMczHxcReeW1c4Xn+FLi0vJWSog6krJBgAJJEpGw4cyeNNMmKXFEJKgCZjxpv9UGORO/vCXkbOheOKNavHISJaUoEndZkny9qXrxZWtKTpUon4l3gTy2H9vSiW8nJWFKmLggkn1n1615j8CUgRa51HnBm3750jbKxRXmmFccUnikCAm4BJ3OkXM25cedSOD371RUUgS2gwLWg6RbYDw8t6rTmRbKgDKtp4Jnn14CpYbPkI8IiR8RtM8vl9PKhaNoKrBYi6g0dHJKQCfIEg7zvvzNMsrccSP5K0cyna34kSTPUdN6DczYq8W5JlNiT5p5f386sYzlLRvA07pbAuYg6lAbyTYE+nEZZnZqcPjDHiAUOY4eYoogESk2rOZZnyHDqWtCT0kelwCaaZbjEmwETJCSItO4MmRveqQ8lEpwtEnaiU0Q+gSIG/K4qp0Wrp7Oaqwh31WIVQ2iiWU0LNRALqzQk0NU0qoi2XDCCavS1XzKpq2lcdKxeEkCvV16DaqlrrIDZWsV4lFfBVTFMKD47FJabWtWyUlR9BMeZ2rhPaEHFYhTrjrTaTso6pUOMQkkidhuBpm9bntVnOKb1JIDmskhkpb0hoKOk7lS5ABkxF+UHP5bhl4hzQG1oMFSgSyG0jiZSmY6WNRlK3R0QVIEy3s0y4AlslcWUqJj0+ED51quzvZ9wEaVq0C3G4HOd44UOhatQZRpDSTfTq8XVR4+tdAyZxsIATeN7HfzqX+nRVKlZ9g8EUDrb3pJ2keKZHn871riLVnM9wiVEbTzO/8AtTuOYZPTnDmPUFhMmQSo/wBUGK8wyT4bXJJvw4+/HypvjMuGsERP6UXgsuTIkX4dKj7LcfYPhFhJAJBAEH8MfmfO1PVsNrTPGLwB+X5ivBkExM78P3anuFywAAAX9/8AajxYL9nP87yN0DU2qRy3nyvPvagE9rClSELQklACROvUIPAggCABx3mur4TJUpBBMk7z+9qyPbjsmgoLoTtcxI+lZQaEk0+hhkuflbjaFpSErB0kKm8EwQSTJuZrQuiuadms1Q2jQ46VN8FzpW2QRHLUmYvFjvIJroWWY1D7SXEnffaQQYIMW3G4sdxY10eL4cvlXsIQ3VsCqxVoFVolYFoqSWqlVrVNQKLm27VKamE2qlVJJjwLSaoUatUbUOo0DMmgVTm+ILbK1p3SknadunGiGzQfaF7Rh3FEEgJJISFEkcgEmZO1udZvDJac17S49S16kYltLUQpStMqMmdBt12PXbYzsmsKQpSlher8R/mEcN7gX3MDkd6R5tnTTilKLSElNtLh0qSBwIO8f5VE9KB7P48F9OuRJ0gIAIAEGI2IuDHE1BYy3aNtiLKJKIEbKIgDhJTYjypjl2JWIPepgcAkCPdYpNjXRquLTxUTJ6jYeVqHDm//AC6FkX8QRP8A8zNTupHTHYG7cx6iLK5SYn2ifrSXHOyoyVyeJTA/SkAzhpBE4IIPLu0k+hSAK8cxeu6WmW+ICioK/wDjce1NOeDQjulmKx5QsBV/SmmAxBJsOtJ3nXdJSplpcCfA54oiQbwT/aq8FmaENrWpJBCbeMzO0Gpxe2Wl1SN63i7JH7FNMM9NcwwvaPWIQ2tSxwubeYrQ4PPvCFONFBFjLqJ/9uon5VaPkTOeUGboKoXMmgttQ5gilOHzlKo3AP3tUfLjTB3EeEiQTG/A07kmifBpnFs9yjunVaQhWrwwVpGnVYeGZnlffha+p+y3Ulb7fi0pS3MmRr8W3+kgddE0u7ZrJIeSZCTp0kAEHoYkiw42mtP9m2j/AAgKUwpUFZO5USflATA4SetZLUSl0zUHerAsVQ+qqEu1Y5rLQasbNUk1JBpkAZtbVU4mKo/xEWr5L01KeFYFyxaqQmrdVqiBQiwnqBFL+0T0MmRKPv8AKINyeAmL+9r0W+qBS1WOgwaLYidOzl2aZW0qXA3NpE95KdxbxEESDfSBw4UnyTBlb4AA0oO19JkzE2nYk38+Vajt7gQ040WUiHdStCRpgoglSSkjSYUdo48zOJx2OXPdIStJPxAhIJ4n4d558vOoYdSVpP6dEzXGpcKEtlKiAB4YCR0A3AHmaAxK0hRBK1K492QAPMx9av7PYG2tcDw78Bb6AfSmD7zejQNDaZsNOpZ6quI+dS7kdCXFClrEKKSjuSQdiXGSfmYn0pZicEZg4d0dSo8P+3+lGYjBJUSpp9HTwqBPmJPvU2sNiUI8biJ30xOxtNwBWYYySK8GWVqS2Xlsu3AOorTH+ZK/EPQ0OnBraccaew5c1ElK0rhKp+FQlUAdKPcxSXNIebCgmTpVJCkkXLaz4krF7EwaEbUe88KG1NzCe8UqQOA0qkjyv50WOnZ8jE9wpKVd2lMj4lgxykDbjxrV4BGtMtOBaSfvYdbiUnorltvWcViUpcKobSNQACBA3vHh53rYZLmrqwRpJ/Dq0wscgobH0rKkKyxnDPfeaZcH4m9I23lKgI+dEOo8JKQUi8pNr8o5eVF4dRUop7woURKQoX9z8Q/c0biMJLZ1wTBmNjTpfCcpHJe1zykd0EgXWFEzsJgBQ5G8njblW87E4MsYJpKlFS1aYmLA3CR0Akn15VyjKEK/xKluArAUrwqNieA5ECxjjauq9nMvdQrUo6Z/ErUu94AMhA6CqrWcssWj3EpP9+tBpNF4m9DhFWOcJivpirUoqDyaLdGqwcUXh2uNUtNXpklMCpdsqsRSoVNFUOrvXyVUF2YtdRIpPjMDJ6dKcaqFzRKu6X3YlelWkc1RYb86aWoVo5721WlMJJAcZQtYuTAc8Ik8T4TWI7NshzEoUQTIJkm1oFvWtPmveult5xA0uuJQtKTqgKTACzwhBQU8DqUq2qAAnKjhMQy2DqAStOoCNUqJ26Sj3rmlSbZ2wS/NL4anHhKGFnVEDxAcp2rFPYx5xyQSEnYJk6p3sN62WY4YutFPEi2832uL1l8d2cKEx3ywYgpQoDrE+fA0ijYzYrxTuK2CzE7KTHtalr2MfQZWbdNq8VksOj+IsAnZQInzMxVWb4I6iULgHdN9PpJNU4L0bn9Q8yvtChYDajf897VJeFWGtYUT4k28jvWVyTLVKdTExqH1rsGHycOYBQSPFf3mlad4PF0jDvZ9AEC6ZvaZmaLyftg/cFuQTcpFtXMXsZ4isjmuWLZVYmeMi4P0o3JstxaiNLw0WJgi3+ki1H87QrmjqmU50+sAr8UnjBHsRWzyrEl1tU2vHuJ/OubdmsLje8BUlp1u06CUqEcfw6ugrf8AZ5tSe9KhAUrV+X5UYRcXok2msMJleDa75ClBQ8birAWJn5RbzI8q3+AwaUAqAAKgIHJI/Ukn16Vnuy+HCnFLIt4lgSRp1K8Ij+mDHUWrUrXNV8K9nP52r4lThq7Dt1WmiUVcgfA15E1QrEDgatYVNLIMAzDogVY9tUUGvHDNKUbATvV6GxVRsatbXSVTAWFFVPN6kqTMSCJHCRFqtKhUJp7AYztFkip71BA0gQkyI0CUgqBCVARbUDE2isC9iF60lS1KUFFSUG3dp1eK24MxbhFdfzg+AgGOpEiONpE2njXPsblJRqWpaFfeCykd8fvlK1fe2HtsKh5I/C/h8tJxZbjMcG4MTYb7RwoTG48vpA1AbeEABAnpxMc58qDzAkpSqZCQoT1B/vQeFgmT7c6nZ0pFDuWLkwAesVD/AIWZHeGf8o4edaBOJAHAUoezP+IlpA1OOKCR0nj5AXoofjgdgcElABSLi56VvOx/iZWItqPrIvSHCZapCSCCTzjc/wC1afsizoaIiDqMj2p4ok9TM/2pyNhw6FeFSh4VdazuCyosuaHMOlyOOkG3Dka1/a9YPgNliFpjiDt6yKjk+PS8gBRuNlDfhS1tFl1Ydk3hTKW9IPAGDE+XDlEfm6xGKSGVHcKhMgXJVa44Ulx+KUBwPUbg/iFNMJhNbISo3PiJ5Ha3tRi7eEZpJWyvBBGpxSPvkKJ68vQRRFTOFDaYFh+zVIXXVFUjz/I7lZe0miEJoVpVFpNEBm8KVGnuDatXjOEAowCKktZZ0i1Ca+e2r1tVReNOKBLF6mlNeK3r6psxYKn3VUtqvRopksMJMyZMVmM7w8ITOxVH68ReOtbXHJFZftx3YwwC1QpTiA3zK52H+nVU2rdIEf8ARi3X0aSlcgAnxATz48SefKgEMggaZuBciJJq3EOAoUlQkQIAIBQZ5+nLjSnJndE6lXTIgzvMDjYedQPQToqzLGlu1JstfeQ+H0jxJkgnhIIpkEd6tS1QEgk34xxBO43rx91BEiRFvMR/vT1RpeT0h5hu374SoL0648wRzT1p/wBhu3Golp0cCQr6zXLcWglWrkmYG/Un0v6077Kv6ZBTfULmxEcPO5rXSsXmqqjY9vMY++hGIS0Ww2VwCfGpswQopB8OxMbwaW9lMzkyLzuOv7NaQuoU2AYKtFoNiCYE8vO/pWSyzDow+N0TCFmY5G0R8/3sj3S0JpYb0IVA63E8hv141scpUFIkbbD0FZnEuhWhP4IMjjPCBw6itXlg/hI8vlw+UVXxdnN55WijHGgRTDHN8aABrqOAvaFECqGavisEvTXyq+TX03qaRVssaNQdNWpFDvijYrRFsXqeiaoamaIDvDjQ7Cfd1U1LgVWkcfr5VViFWrN0jGa7ZdqUYVInxLUfCnoLknpasVnmaHGMatlMuJeSBJJQAdWwJNuHCKSfaFmAed71K9Q7xSE3kEIA+p1e9ZrC526w9qVqbKL6CCkkcJB4U8EorfYvFvTTuqTKyFXUQpMXBCrgCD0FL8qcGt1BAsAom1zPr1/YobEY8OfxWYDYMrbO7RO5HHuzz4Wm0VLDJJfBSQSsGTNhNpCum1c0ocWdcZ2hZmD5XIT4QIEcucedGYHJFlI8Zgi4ImPTej1ZeZUI24fEYHMgbz8qIdw7iUHRKVx0536H+9BseLB2MlemAtpU28Upty48zWkw/ZTEFOodwAbnxq2jjCbnb5Vl2MYsrSkpTZVySZkgjebTv6VqsmzZ1ACUN/F95SrAAbj9zW5RS1Dck8oCzrKsWygKSpEC9pkhPC+9DdmXlP4hAcABAkTMqgQR03sfea3pSp5MLF4NlWg8QOE9PKs/hcsOHdUUjVIIB4i2mE2m87Vkv4T5UzTZY0VrIjVKo8xa5PQT6VvGUwAOQivz92s7SLwelhpWl6UqcUgABEQUpSTx2UfQV2DsH2gOMwiHVphfwq5EgDxDkCCDHDaqwjSJTlbH2ITIpOsQaerFKMY1eqo55LTxg0UmhGBRyU2ogPVKqAVXpE1Y23SFWWNGvHBNfCvlGsAEW3VL+ZMNLQ2txCVuToSogKVG8dP1pP2i7b4PCFSVuanB/wBNsajJkgE/CgmDZRFgTXHsyz5nHY1559SYShPcMrcKG16VBICnAmbBS1wIJMieeUfpv4jrGedvWWVhttjE4lR/9O0VJvt4iQDM7iRQ+b5u66ypvENnBIcTBW442pzSTBSltBMKUJEk2+VcpczNaVOKw7rTaZUgrwqCFKvqu4oBarnfYwau7ll/wuSXlTHfYhRCTsBpA8KuPimbi1qdQQG2EZ/kyV2wiylLJU6nWFrLigADoWhJSSCnZJUZmYpOc0TiUf4fHiHB/Le+8k8lHiKcZT2fxTa22ziCGm1SptJWEjdREfCT41D/AFGie2GXYdUyNStMCCLQbGLxudom1MlgrlbowmKy7EYRetJkC4Um4I/MUVlecpKgoQlf4PuHq3+E/wCX25VFWNdw38NR1tkSLzpBvvtMbjhS/H4NC/Gzx4fvY0kopoom0dQ7MuIdBM9CDrMDjcxx4A84ii8cynxDcC0cRxMnebzyEHjXKcg7SPYVfSbg7++9a5Pa1DidU3nmJ2v4fM7/AFqLh8KxlTGiMsYWVKPC9plRO/p/bnWzwmVNFCUpGka9vpHSOFc0YzwWIVfSdcERqgkiDzn61sck7R3SCZBgAJkkQFDhwGoR5ilUS0vMkjVjChCSsqBgGeekDxX9d+vpS3FuNrnSQG0gkkSZNwInz8jVeZY1Tza20TMkKJ20k3iOHhEVku1mYHDYVaEW1jSOKiTx2sTxpnXRFJydvo512kzPv33HgPCpUegsj10j5Vv/ALCswdVjlNhxXdd0olGo6ZkQQnYG+4rk5c8On92rf/YespzEKmB3ax5zFv3yqqROTw/TIofENTShvtbhe9Wyt0NON/El3wiIkFKj4VAgg70ozT7SsEy6plXeFaeSQArlpKiAQeB2POsrTA6aNMhqKuArG5V9puBec7pRWyuYh5ITfzBIHrWwQ4DcXnlRJ8SzTS/tBnKMIwt5dwmAAN1KJgAfvaaYLVXJvtZzgOuNYVCgUolbkGQFnwpBPMDVb/NQQ7dER9rbqXSFYYKbg2QVakciVGQR6DpyNL3bXG4tSktkMtiytABIngXSbKj8N+YFc970KdW3xU2oCSYOmCJ9qW4vGuugJ7wBKUyEIMBIFySlPhSOqiKLAk2dCTlzSEFB0kKOpSSWtKlgEAkBG8Eid79aW4rJ8MJjDFB/E24k9LBVqwWBfc1QglRF4PEDeKfYLMlLTIVflJt5VjcWN8sfwjQLai8QZstA0gmJsnjYXjhwqvMMKwR/y7SVaoSFJTIRO6jbwkCYnjFIcVme4JJPH9DVmXY2VCDoVwI+hHEUeVGcfZpsDmLiAnDYlxZbBlt0blIsEuXAVANiTym0ELMwdVBUoKAkgE8xwnaYi1GF1LqC24QDzix8qBfwbiUpS+FraTJC21biIBIMgKFrkXAiZApk10DjtiZWKOlSeCrGehBsdxccPLY0Gpg61dxqKQNRBiYA8RIFoqWKZWkAkEBQ1JO0pmJHS1BhJPOlool9JKdC/iF+dUOYci+46UUdKtCVAJgwpaZJgncibkTwj4R51BGsFRRK0puSAbDmbWHnSmqiOGUU+LcAgHmDuJ84PtXSOyuZNBAVpItwve8b7b1hMFjGzKViyhBPLjI6ggH0rRZSyWk6gkutmPG1BjgNad0Drt1pJRY8Wr06ExmYBsBwuOHTnWR7TLOIf0gju8OkuLKoKdZEAGSNW4HrXmY52oDum9Ou8BJStQEXlSSUpPmTte16H7SLbwuBSzq1PPAa9JSYWSFOFSgJVbw7xCkxxNSUaoq5qqOfstFSkpG5geprV9lC43jWO74PBsxG9gqfQ0gwC9IKtMqPwn8On4jHy9K232T5aXMUFm6G1d4TyVphPXia6Y9nNLpnWO2PZIYpsutgJxKE+BekFRABhN4/EY5GCCK5V2iS46tTbyEd80EogoDetHxA2J0LJUTKTEk+ndsmx3eLc0/AjwAwbrHxwelh5zyrjv2xYpv/ABQW2QFoAbWIIJmSCOYFx/qpq0mnioymPykhtlThUlKwO7dsS2dwh0iNQ5WBF6uw+ZrH8J9xxtaBAUhW6eA5KSZBBp0y4l3CNpc+BcpVA2UDKVDy/Ks72mwKgy0rT/EaJYdgEz99tXkpJkHrQCneM6b9ofbxRC8Pg9rocxH3RwKW/wAR4ah6c65zh0lMX4RPP9ONeB8hWlV1ESRwQny/EbW4CpoUZ3tasarFrjX/ADOrkKX59iFpKmphBOoCAJvME8YP5U7WIWfShMzQlYlQ+H6UGOhJk72l1J2tHrWofy0OfxWSEr4jgr9D9ayOFbBUBMXt+VNkYtxhQ4poWF/wg+0pSilSSlSdwd/7iqcOQFQoXrRozxp0Q4gHqfiHkd6irIW1gqbUVA7XuP1rV7AmVYXSY8R+fh9aZYbGlH3p6g/lWfdYfatuPn61NL1vFvRA0jSPYdh4eIFJ5t7Tz07XgTSTNezTqAVoPeIF5SDqA6p39poUZgobX9/y3pvl/aXRYn0rXRqMqGr1cWFz4ZBO8Hfz51tP8Vg8R8bYCj95HhVPXgfaisNlLabtqB6L399vpTKr0HJpGAxGWKCU+GCDc3uD0nh0qOXZm9hnJaWQRymD6HhW/Tlyy540wN/Py50wxXZBp6PBCjxFNw9oX9V7M9l/anBOJIfYDLhiXGUpE9SNJA9qqb7MYF0/w8wtwC0iRxjf8htTbGfZSvQpTawYuBFJMF2AfUoa0qQgfEuNW+xCAJig19QVJemO8q7MZawQcRi0ugWCRCRvN4MmtZludpe/hYBrShMBx1CUwhJMWBjWqxMdKz3Z37PEatTkrTqUEhYg6QTpOmYkiDB2rpGWYFtptCQAChOgEhIOkbCwFrD2vJpkicpL7YwybDpZaCUgCBBInxabajJ+I8TxNcG+0PFl3HOgaQVHQNW0kQDtY9eFda7XdrE4Vo2lZB0+fCuFpUrEYgkrImSpUAnqACRc+dCWIaG6aNxBZw6kkXQUkiOSgk+XxVpsqa78JdQ4AvQEq1CUqSk+E7iFCSOo8qy/abFo7gpRM6EBXM6VjeOYHyrYdlcApvDIFpgSPwnlSGf05bhn4WUgyd1He3nTRCqzuQ3WryH1p4DestKsseEkjmKXPmmDjl09ZHymgMYIJoMVgWJagICBKpPsTt5UTIVLLkBxJjnfz51blrwCkKiVCY/q2EjiJg3/AA0kxaCFKVN9VjxJ3J/fOsMXPYcoVfbmKMwuPcbgoMx7e1U4XMUqGh30V+tRxOFUPEggp6XFLVaMP2e06VQFp871J/EYUjVZU8Bv7Vle9B3F6KwbkG1ZSFa+DF0JupLao4RwPUcKWvSo3Tp604w2ICrDcb3g0YrBoULn3/Wi1emszHjT8JtRLGduIsZozF5KsGUGRS3E4VwDxI9RS00FMc4HtY62QUqlPFC7pPH08xW5yL7QMOuzqC2o8jY+RrkCmjyPtVcqHOnU2hZQjI/UGVZzh3EgIdTqiY2PsaaMLBMG4ivythc1db+FRHlTjB9s8Qj/AKivf86b9Cf4/D9J4ltIE8ulZ3tBmvdtgo8apgAXPtXJh9oKzEqX6kke39qKZ7eogBQFuQEe0U36C/lp9neWY7FKK1NqAglOoiNpAjeTtVuX9n1ttwhpOom5UsEg7gBQAJjy40Sj7RWhMIJtsAKU5r28UsylvTPM0jd9jpS6DmMtwzEHEKKlkEnSfCk6geIvc9NqrzPtelohLSjHQCfW9J8IXMRJcgpMR7zSLEthTitNxJjyBgVg0vYTliAgJ5qk+g/3pk8KW4hYSVT91AA9Tf6Ue9zoLB2UYtcd2eSx9DV+YtwQeBFBZsqGwf8AMKOeGtsEVjUhXgl6QroTVGYLStOoC5PyqKVnxgC52HXammXBKFkWPdoIPGVq39pj0orQPNE2cMhC0gcUJJ8zVDGIWgAg2PDgaY4/D6+PiHPkbj0pfi5ASkjYQPqaV9jrQk4lC9wEqqCmlD4dulL6uZeUNjQMXLdXxJFepxix94+9ef4yfiSDU2XEzYD1NYwZh88Wncn99KI/49PxJB8gR+dD9yk7wOtTby5JP6G1FWB0XLzRk30e1VF5lX/TI9d6vGUp4VJWSHhW0GAyWsObbH1NWN5S0fvE+VROAUg23olskcxR0z/hFORIO2r1iiWezYN7xUmH4+971cvtAE21ewrVQNLm8hbqRwjTRKiglIKUki4keKdtrx5g0N/x2dkqiLk2EeVRxuPACUtKKgtI8JGx6nje9bDUQxmYJgttDSjawuTwAHrR2S5Y0ynvMQQFK8ISTYDf3r7CYJDKNbglwiQI48PKqcSjvVlTu2yUjZI/WjeCmXcf1urPMED02p65tWawp/ie/wBK0pG9BaPIBzGS0rpB9jReTr1NEdKGxI/huDpVfZt+DB8vesb0UglDpPEXHmL/AJVHDLgeFJk8YiRb0N596LzJqHff5ihMJhHtKVNklBuYNgdjIrGWoPLZN433/vQ2bYZIQkzc3jlwqaEuqSEm0SCeNqtz4gNsg7xPWOE870cMsM2pMVbhUyqOhrx1BB86vwDRmY4GPOw/OkXYwMWzUQmnSmxe3E0N3AIVHI+5tTUCwBKyKIaxyxxp/icoDuogpTBOxMzuZTEQelIn8vWi8SOJE+u4pXFoNl7WbLnePKmGHx5kEqnpNZ8jpUkrisZ6apONb4yD7g1866FfCL9az6MVzNFs48jiI61uQvFjVnClXxkeVEnAptDdqBw+ap+8RRD+bpiEuCeE8P0pkChjlyW4UoxCoASRfQJgpPnqNL8Fh0pcU85ZP3Bz9PT51cxnEN920CsW0iPhAFwTVSsA64rW4tKQNhNh5Afu1E3RN7G65PHlVTOFddnTMDibUW2lhuLhXOfyFWjMZs2LeRAogv4YbBCXPRX0NaFRvWfy7+Yn1+hp86rxedIhpHjwsocwaS5SshdPHE2pBgzDnvWZomjzROotrHGxpNlWIgqQeMkdCm9h1p0gy0RvBBrMlel0nko/WmsyNbikjvUAbOaY9YBpdnEpccKx8J0geWwFX5rjAO4X+Agj+nf6g0XmrLLiw53o7s+K+4nhHE0asVCPJsmViF3MJ3KjwFe4x1IcKUfCiwMC/E7eQpo5mwUO5YSUo5/eUeZ/Ss8lMTzk/X+3zpUM+rGSBIqphA1dPD9dX0qkTAANzR+Gw6lKSEAajMSUjewuTG1MxRjhIJBBMx4piJnh0iN+tN3MAHNO1vPjFxBF7cZ3rO4XEqUdSoBO8AJH/tFga0eX4gfKmi/Qs79GfdytAJIEoJM6fiSAYKkwZ02uncb0crs4yrShJ1FQBCwRBmdPEFXwnYW4xIpv/wAM71Q7sCyiYmCTEAyORg+kVblSVNunQgKuQpuUyog6Spvi07aQkwFj0g8aFcrWCPHfZ28keFSVGJ08Y/Wszi8hfbElBiuu4DFlDaVFZXqMDcnWkSoFOkFBgFWi8DiYks2ylaLgEb3H7/Zo8IsX9ZLs4B3ahwqaVEfdmu1v5FhlyO7AJ+8BcUlxXZdopJlIImVRKRpOq4VMiAoQZtU34qKR8yZzI5i4BEx5VFOPXzNdJX9nyHUkglKryNJsRvKTBEcqRY77P3mySr4BcqAmBxtW/N+gryxMqMernXwxRP3j71o8N2QKlEGSkCSpMbKOhBMmASo7ckHnWhyf7LC4JdcKOQSJ+tDgxn5Io5xlp/ip9foaduXNJMt/mp8z9DTo0sXoZE3T4az7SocB61oDtSBf8z1rMyw0mAckxwINZrMBDi/M0+y7dPpSTNf5q/P8q3o0NIv4oqSkE7VUgk1XVjVBjDDC4jQJHxXA6SIn0FVtCPKqk7+/5UUmmiaQQykFQjeD8/386k6W1hRlQUnSEgAFJjeTMg/EduVV4T4Xf6Ff+SKi7w/fA1mJHWWMzMzTZl02IpWxtTNj4aosEka3I8eEonY8DHvTjEK79sJhMyCZkagNwSkggeR5Vksv/ljzrU5f8KfIVSznf0XOpIXpV4lyASSQl2x0pUr7q41aVf1AyCaMwzkBS76ZIhRujbwqT91QvNzMgi1Ls/8A54/7X/6NOGP5zv8AThP/ACNAa7DsI+QLix41ThngSpIAMkahB3Mpi9oOqbc6Me4/1L+tBt/F6o/+xNF9CrsP7PvkNp8ASIiIUNNtoIHlYRyJEGrsxe1mEmNCgpwEGFIIgpnibgxxtzr3DbDyNAp//rV/Qz/5ms+gLZBOTYaCVhICSSFp2AKYCUpEXCQCNxB1b8HDS78No+lKMo3xX/dT/wDWimCKKNL2f//Z',

                                'https://upload.wikimedia.org/wikipedia/en/thumb/d/d5/Jorah_Mormont-Iain_Glen.jpg/220px-Jorah_Mormont-Iain_Glen.jpg',

                                'https://vignette.wikia.nocookie.net/gameofthrones/images/2/26/Winterfell_ep_Jorah_s8.jpg/revision/latest?cb=20190418013520']

dict_faces['Daenerys Targaryen'] = ['https://pmctvline2.files.wordpress.com/2019/04/game-of-thrones-daenerys-targaryen-dead-season-8.png?w=620&h=420&crop=1',

                                    'https://theartofgrowthandofeverything.files.wordpress.com/2018/06/daenerys-targaryen6.jpg?w=540',

                                    'https://res.cloudinary.com/jerrick/image/upload/fl_progressive,q_auto,w_1024/jomkgr0az1tzqkbjry51.jpg',

                                    'https://www.thesun.co.uk/wp-content/uploads/2017/08/daenerys-targaryen-1280.jpg',

                                    'https://fsmedia.imgix.net/ab/d9/98/b5/dc04/4c58/a334/f2c1ec6a3e5f/daenerys-targaryen.jpeg?rect=0%2C151%2C3150%2C1576&auto=format%2Ccompress&dpr=2&w=650',

                                    'https://www.washingtonpost.com/wp-apps/imrs.php?src=https://arc-anglerfish-washpost-prod-washpost.s3.amazonaws.com/public/IL5JQLXUWFGTVF47RKOGFDNBNA.jpg&w=767',

                                    'https://i.guim.co.uk/img/media/02f5315a3ddd09325377357e22f052a9a6759e1e/0_249_4000_2400/master/4000.jpg?width=300&quality=85&auto=format&fit=max&s=e44acb69590163711247bb5a16e44f97',

                                    'https://vignette.wikia.nocookie.net/gameofthrones/images/e/ee/QueenDaenerysTargaryenIronThrone.PNG/revision/latest?cb=20190520173137']
# Check urls and print last image

# if any image causes problems (like error 403 forbidden) then remove it and choose another

for p in dict_faces.keys():

    urls = dict_faces[p]

    for url2read in urls:

        print('\rChecking: %s' % url2read[:100], ' '*100, end='')

        img = read_image_from_url(url2read)

plt.imshow(img);
os.chdir("/kaggle/working/")

!wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

!ls
#Create the haar cascade

face_cascade = cv2.CascadeClassifier('/kaggle/working/haarcascade_frontalface_default.xml')  # if not running on kaggle, remove the initial '/'



def find_faces_in_image(orig_img, scaleFactor, minNeighbors, minSize, maxSize):

    orig_img_copy = orig_img.copy()

    gray = cv2.cvtColor(orig_img_copy, cv2.COLOR_BGR2GRAY)

    #plt.imshow(gray) 

    

    # Detect faces in the image

    faces = face_cascade.detectMultiScale(

        gray,           

        scaleFactor=scaleFactor, 

        minNeighbors=minNeighbors,  

        minSize=minSize, 

        maxSize=maxSize 

    )

    

    print("Found {0} faces!".format(len(faces)))



    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:

        cv2.rectangle(orig_img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)



    plt.imshow(orig_img_copy)
scaleFactor = 1.3

minNeighbors = 5

minSize = (60, 60)   

maxSize = (70, 70)

find_faces_in_image(img, scaleFactor, minNeighbors, minSize, maxSize)
def find_faces_in_frame_of_video(orig_img, scaleFactor, minNeighbors, minSize, maxSize):

    """Find faces in an image and returns the bounding boxes of them"""

    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        

    # Detect faces in the image

    faces = face_cascade.detectMultiScale(

        gray,           

        scaleFactor=scaleFactor, 

        minNeighbors=minNeighbors,  

        minSize=minSize, 

        maxSize=maxSize 

    )

    return faces
def video_file_recog_haar(src_filename, output_filename='output_haar.mp4', framerate=None, scaleFactor=1.3, minNeighbors=5, minSize=60, maxSize=70):

    print("[INFO] Reading video file...")

    if glob.glob(src_filename):

        vs = cv2.VideoCapture(src_filename); #get input from file

    else:

        print("file does not exist")

        return

    

    print("[INFO] Initializing video writer...")

    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # DIVX, XVID, MJPG, X264, WMV1, WMV2, mp4v

    if framerate is None:

        framerate = vs.get(cv2.CAP_PROP_FPS) # use same as input video, or can set to 20.0 / 30.0

    out = cv2.VideoWriter(output_filename, fourcc, framerate, (frame_width,frame_height))

    

    recog_list = []

    frame_counter = 0

    t0 = time.time()

    while True:        

        ret, frame = vs.read();

        if ret:

            frame_counter += 1

            if frame_counter%(30/framerate)==0:

                min_face_size = 60 #min face size is set to 60x60

                rects = find_faces_in_frame_of_video(frame, scaleFactor, minNeighbors, (minSize, minSize), (maxSize,maxSize))

                print("\rNumber of faces found in frame " + str(frame_counter) + ":",len(rects), end='')

                aligns = []

                positions = []

                for (i, rect) in enumerate(rects):

                    draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), (255,255,255), 1, 10, 10)

                    cv2.putText(frame,"Unknown",

                                        (rect[0]-4,rect[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,

                                        (255,255,255),1,cv2.LINE_AA)





                out.write(frame)

        else:  # end of video, no more frames

            break

    

    elapsed_time = time.time() - t0

    print()

    print("[exp msg] elapsed time for going over the video: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    vs.release()

    out.release()

    cv2.destroyAllWindows()

    print("Done")
video_file_recog_haar(src_filename = get_file_path("Game of Thrones 7x07 - Epic Daenerys Dragonpit Entrance.mp4"), framerate=10)
#Global Variables

person_embeddings = None
def augment_image(img):

    aug_images = []

    flip_img = cv2.flip(img, 1)  #https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=flip#cv2.flip

    # for more example see https://github.com/aleju/imgaug

    aug_images.append(flip_img)

    return(aug_images)





def get_person_imgs(urls, min_face_size=40):

    """

    Given a list of URLs of a person images, this function will extract and align the faces within the image and detect its pose (left/right/center)

    """

    person_imgs = {"Left" : [], "Right": [], "Center": []};

    person_imgs_count = {"Left" : 0, "Right": 0, "Center": 0};

    

    counter_break = 0

    while True:    

        for url2read in urls:

            #print(file)

            #ret, frame = vs.read()

            #img = cv2.imread(file)

            img = read_image_from_url(url2read) # ****** file = url2read

            if img is None:

                print("********************* image was not loaded ***********************")

                continue



            # Augmenting the data - add a flipped version of the image to add more data

            frames = [img]

            frames.extend(augment_image(img))



            for frame in frames:

                if True: #ret:

                    rects, landmarks = face_detect.detect_face(frame, min_face_size)

                    #print("rects", rects)

                    for (i, rect) in enumerate(rects):

                        aligned_frame, pos = aligner.align(160, frame,landmarks[i]);

                        #print(pos)

                        person_imgs_count[pos]+=1

                        if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:

                            person_imgs[pos].append(aligned_frame)

                            #cv2.imshow("Captured face", aligned_frame)

                            #cv2.imwrite("../data2/frame%d.jpg" % count, aligned_frame)

                else:

                    break

            

        if person_imgs_count["Left"] == 0 or person_imgs_count["Right"] == 0 or person_imgs_count["Center"] == 0:

            counter_break+=1

            if counter_break > 0:

                print(person_imgs_count) 

                assert 0==1, "Must get all poses of a face: Left, Right and Center, try adding more images"

                return None

        else:

            break

                            

    print(person_imgs_count)    

    return(person_imgs)  
def extract_embeddings_from_images(min_face_size=40, embeddings_filename='/kaggle/working/facerec_128D.txt'):

    """ 

    Go over all urls, extract and align faces, feed each face to the embeddings net,

    and saves an embedding vector for each person-position pair.

    Save all embeddings to a .txt file

    """

    print()

    print("[INFO] Extracting data from images ...")

    data_set = dict()



    for new_name in dict_faces.keys():

        person_features = {"Left" : [], "Right": [], "Center": []};

        print("Extracting:", new_name)

        print("number of img files:",len(dict_faces[new_name]))

        person_imgs = get_person_imgs(dict_faces[new_name], min_face_size=min_face_size) 

        if person_imgs is None:

            print("extraction of:",new_name, " failed")

            continue

        

        print("extracted person_imgs from:",new_name)

        print("-------------------------------------")



        for pos in person_imgs: # there are some exceptions here, but I'll just leave it as this to keep it simple

            person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]

        data_set[new_name] = person_features;

    

    global person_embeddings

    person_embeddings = data_set

    with open(embeddings_filename, 'w+') as f:

        f.write(json.dumps(data_set))

    



def load_embeddings_from_file(embeddings_filename='/kaggle/working/facerec_128D.txt'):

    global person_embeddings

    with open(embeddings_filename, 'r') as f:

        person_embeddings = json.loads(f.read());



        

def identifyPerson(features_arr, position, thres = 0.6, percent_thres = 70):

    '''

    :param features_arr: a list of 128d Features of a face

    :param position: face position types (Left/Right/Center)

    :param thres: distance threshold

    :param percent_thres : minimum confidence required to identify a person

    :return: tuple of person name and confidence of detection

    '''

    assert person_embeddings is not None, "Must load or extract persons embeddings in order to recgonize persons"

    result = "Unknown"

    smallest = sys.maxsize  # initialize with a large number

    for person in person_embeddings.keys():

        person_data = person_embeddings[person][position]

        for data in person_data:  # in our case there's only one embedding per person-position pair

            distance = scipy.spatial.distance.euclidean(data, features_arr)  # same as: np.sqrt(np.sum(np.square(data-features_arr)))

            #distance = scipy.spatial.distance.cosine(data, features_arr)  # if using cosine distance it is recommended to lower the thres to ~0.4

            

            if(distance < smallest):

                smallest = distance

                result = person

    percentage =  min(100, 100 * thres / smallest)

    if percentage <= percent_thres:

        result = "Unknown (%s)" % result.split(' ')[0]  # show highest score person for debug purposes

    return (result, percentage)
model_path = '../input/model-20170512-110547.ckpt-250000' 

os.chdir("/kaggle/input/")  # if not on kaggle environment then omit the initial '/'
# initalize

FRGraph = FaceRecGraph();

aligner = AlignCustom();

extract_feature = FaceFeature(FRGraph, model_path = model_path);

face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
#load_embeddings_from_file()

extract_embeddings_from_images(min_face_size=40)
url = 'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/best-game-of-thrones-season-8-fan-theories-1554917935.jpg'

frame = read_image_from_url(url)

rects, landmarks = face_detect.detect_face(frame, minsize=40);  # min face size is set to 80x80



for rect in rects:

    draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), (255,255,255), 2, 10, 10)

    

plt.subplots(figsize=(15,10))

plt.imshow(frame);
idx = 0

rect = rects[idx]

plt.imshow(frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])

for k in range(int(len(landmarks[idx]) / 2)):

    plt.plot(landmarks[idx][k]-rect[0], landmarks[idx][k+5]-rect[1], 'r+', markersize=20)
for (i, rect) in enumerate(rects):

    aligned_frame, pos = aligner.align(160, frame, landmarks[i])

    plt.subplot(121)

    plt.imshow(frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])

    plt.title('Original')

    plt.subplot(122)

    plt.imshow(aligned_frame)

    plt.title('Aligned')

    plt.suptitle('Position: %s' % pos)

    plt.show()
features_vector = extract_feature.get_features([aligned_frame])
features_vector.shape
features_vector
def frame_face_recog(frame, min_face_size=80, percent_thres = 70, verbose=False):

    """ 

    Detect faces in a frame, try to recgonize them, and draws a box around the face with predicted person + % confidence

    :param frame : the frame to indentify faces in. an array with shape of width X height X channels

    :param min_face_size : minimum size of face to detect. integer. e.g: value of 80 is set to 80x80 pixels

    :param verbose : True to print debug information while running

    

    Alters inplace the frame with predictions annotations

    returns a list of (person,confidence) tuples

    """

    aligner_resize_to = 160  # the aligner function will rescale image to X by X pixels before sent to be embedded.

    

    # Detect all faces in frame and get their bounding-rectangles and landmarks

    rects, landmarks = face_detect.detect_face(frame, min_face_size);

    

    # Go through each face in frame and perform:

    #  1) align the face (using aligner.align() function). remember that aligner.align() returns the aligned-face and face-pose (left/right/center)

    #  2) extract aligned face features (embeddings)

    #  3) find the person the face belongs to

    #  4) draw a box around the face and label the person

    recog_list = []

    for (i, rect) in enumerate(rects):

        if verbose: print('BBox %i:' % i, end=' ')

        

        # align the face

        aligned_face, face_pos = aligner.align(aligner_resize_to, frame, landmarks[i])

        

        if not (aligned_face.shape[0] == aligned_face.shape[1] == aligner_resize_to):

            if verbose: print("Align face failed!", end=' ') #log

            continue

            

        features_arr = extract_feature.get_features([aligned_face])

        recog_data = identifyPerson(features_arr, face_pos, percent_thres = percent_thres)

        recog_list.append(recog_data)

        if verbose: print("recog_data", str(recog_data))

        annotate_face(rect, recog_data, frame)

            

    return recog_list
url = 'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/best-game-of-thrones-season-8-fan-theories-1554917935.jpg'

frame = read_image_from_url(url)

frame_face_recog(frame)

plt.subplots(figsize=(20,15))

plt.imshow(frame);
def video_file_recog(src_filename, output_filename='output.mp4', percent_thres = 70, verbose=False):

    print("[INFO] Reading video file...")

    if glob.glob(src_filename):

        vs = cv2.VideoCapture(src_filename); #get input from file

    else:

        print("file does not exist")

        return

    

    print("[INFO] Initializing video writer...")

    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # DIVX, XVID, MJPG, X264, WMV1, WMV2, mp4v

    framerate = vs.get(cv2.CAP_PROP_FPS) # use same as input video, or can set to 20.0 / 30.0

    out = cv2.VideoWriter(output_filename, fourcc, framerate, (frame_width,frame_height))

    

    recog_list = []

    frame_counter = 0

    t0 = time.time()

    while True:        

        ret, frame = vs.read();

        if ret:

            frame_counter += 1

            print('\rProcessing Frame %i/%i' % (frame_counter, total_frames), end=' ')

            recog_data = frame_face_recog(frame, min_face_size=40, percent_thres=percent_thres, verbose=verbose)

            recog_list.extend(recog_data)

            #cv2.imshow("Frame",frame)

            #cv2.imwrite("../data3/frame%d.jpg" % count, frame)

            out.write(frame)

        else:  # end of video, no more frames

            break

    

    elapsed_time = time.time() - t0

    print()

    print("[exp msg] elapsed time for going over the video: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    vs.release()

    out.release()

    cv2.destroyAllWindows()

    

    known_counter = len([1 for recog in recog_list if recog[1] > percent_thres])

    unknown_counter = len(recog_list) - known_counter

    print("known_counter:", known_counter, "unknown_counter:", unknown_counter)

    

    print()

    print("Done")
video_file_recog(src_filename = get_file_path("Game of Thrones 7x07 - Epic Daenerys Dragonpit Entrance.mp4", percent_thres=70, verbose=False))
# the output file from opencv is just video frames (Without audio). lets add the original audio track to the output movie

# if ffmpeg is not installed then it will return False

#vid2vid_audio_transfer('Game of Thrones 7x07 - Epic Daenerys Dragonpit Entrance.mp4', 'output.mp4', 'output_w_audio.mp4');
!ls ../working/ -ashl
from srganUnified import SRGAN

import tensorflow as tf

os.chdir(cwd)  # reset position to root folder
def infer(x_test, ground_truth, titles, save_results=False, display=True):

    """ x_test should be in shape of (batch_size, 24, 24, 3). images in BGR (not RGB)

    and ground_truth is same as x_test, just (batch_size, 96, 96, 3)

    ground_truth is only used for displaying, and not for inferring.

    """

    x = tf.placeholder(tf.float32, [None, 24, 24, 3])

    is_training = tf.placeholder(tf.bool, [])



    print('Initializing Model')

    model = SRGAN(x, is_training, batch_size=len(x_test), infer=True)



    print('Loading model checkpoint')

    # Restore the SRGAN network

    saver = tf.train.Saver()

    saver.restore(sess, 'srgan_models/epoch60')



    print('Inferring')

    # Infer

    raw = x_test.astype('float32')

    fake = sess.run(

        model.imitation,

        feed_dict={x: raw, is_training: False})

    save_img([raw, fake, ground_truth], ['Input', 'Output', 'Ground Truth'], titles, save=save_results, display=display)

    print('Done')



    

def save_img(imgs, label, titles, save=False, display=True):

    for i in range(len(imgs[0])):

        seq_ = "{0:04d}".format(i+1)

        fig = plt.figure()

        for j, img in enumerate(imgs):

            im = np.uint8((img[i]+1)*127.5)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            fig.add_subplot(1, len(imgs), j+1)

            plt.imshow(im)

            plt.tick_params(labelbottom='off')

            plt.tick_params(labelleft='off')

            plt.gca().get_xaxis().set_ticks_position('none')

            plt.gca().get_yaxis().set_ticks_position('none')

            plt.xlabel(label[j])

            if j==1:

                plt.title(titles[i])

    

        path = os.path.join('result', '{}.jpg'.format(titles[i]))

        if save:

            plt.savefig(path)

        if display:

            plt.show()

        if save:

            plt.close()
def downscale_func(x):

        if len(x.shape)==3:

            x = np.expand_dims(x, axis=0)

        K = 4

        arr = np.zeros([K, K, 3, 3])

        arr[:, :, 0, 0] = 1.0 / K ** 2

        arr[:, :, 1, 1] = 1.0 / K ** 2

        arr[:, :, 2, 2] = 1.0 / K ** 2

        weight = tf.constant(arr, dtype=tf.float32)

        downscaled = tf.nn.conv2d(

            x, weight, strides=[1, K, K, 1], padding='SAME')

        return downscaled





def process_file(filename, downscale='cv2'):

    """ downscale can be 'cv2' / 'conv' 

    """

    img = cv2.imread(filename)

    return process_img_arr(img[:,:,::-1], filename, downscale)





def process_img_arr(img, name, downscale='cv2'):

    face = img[:,:,::-1].copy()

      

    if face.shape[0] > 96:

        ground_truth = cv2.resize(face, (96, 96))

    else:

        ground_truth = face

    

    if downscale=='cv2':

        face = cv2.resize(face, (24, 24))

    elif downscale=='conv':

        gt4conv = cv2.resize(face, (96, 96))

        downs = downscale_func(gt4conv.astype('float32'))

        face = sess.run(downs)

    

    ground_truth = ground_truth / 127.5 - 1

    face = face / 127.5 - 1

    input_ = np.zeros((1, 24, 24, 3))

    input_[0] = face

    

    return input_, ground_truth
face_urls = [

             'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/best-game-of-thrones-season-8-fan-theories-1554917935.jpg',

             'https://i.redd.it/mm9sgp28ri811.jpg',

             'https://cdn.pastemagazine.com/www/articles/CERSEI-LANNISTER-quotes-list.jpg']
def urls2imgarr(urls, crop_faces=True):

    """ 

    Go over all urls, fetch the images, and if crop_faces is True then extract and crop all faces in image.

    returns a list of image arrays (shape of (width, height, 3))

    """

    all_faces = []

    for url in face_urls:

        print('Fetching', url)

        frame = read_image_from_url(url)

        if crop_faces:

            rects, _ = face_detect.detect_face(frame, minsize=40);  # min face size is set to 80x80

            for (i, rect) in enumerate(rects):

                img_arr = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

                all_faces.append(img_arr)

        else:

            all_faces.append(frame)

    return all_faces



all_faces = urls2imgarr(face_urls, crop_faces=True)
tf.reset_default_graph()

sess = tf.Session()

init = tf.global_variables_initializer() 

sess.run(init)
x_test = []

ground_truth = []

titles = []

for (i, img_arr) in enumerate(all_faces):

    x, gt = process_img_arr(img_arr, i, downscale='conv')

    x_test.append(x)

    ground_truth.append(gt)

    titles.append(i)



x_test = np.concatenate(x_test)
x_test.shape
infer(x_test, ground_truth, titles, save_results=False, display=True)

# If running infer throws an error, try going throught the initialization cell once again (tf.reset_default_graph()...)