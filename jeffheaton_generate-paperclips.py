OFFSET = 1 # What is the ID of the first paperclip generated

COUNT  = 25000 # How many paperclips to generate

import PIL

from PIL import Image

from random import randint

import pandas as pd

import requests

from io import BytesIO

import zipfile



print("Generating paperclips")



# Load the paperclip and background images

response_paperclip = requests.get("https://github.com/jeffheaton/present/raw/master/data_files/paperclip.png")

paperclip = Image.open(BytesIO(response_paperclip.content))



response_background = requests.get("https://github.com/jeffheaton/present/raw/master/data_files/background.png")

background = Image.open(BytesIO(response_background.content))



# Resize the background

background = background.resize((256,256),resample=PIL.Image.LANCZOS)



# Keep the apsect ratio of the paperclips when they are scaled

aspect = paperclip.width/paperclip.height

clip_count = []



# We will output to a ZIP file that can be easily downloaded

z = zipfile.ZipFile('/kaggle/working/clips.zip', 'w', zipfile.ZIP_DEFLATED)



for c in range(COUNT):

    render_img = background.copy()

    cnt = randint(0,75)

    clip_count.append(cnt)

    for i in range(cnt):

        a = randint(0,360)

        clip_size = randint(30,60)

        w = int(clip_size*aspect)



        paperclip2 = paperclip.resize((w,clip_size),resample=PIL.Image.LANCZOS)





        x = randint(-int(paperclip2.width/2),background.width - int(paperclip2.width/2))

        y = randint(-int(paperclip2.height/2),background.height - int(paperclip2.height/2))



        paperclip2 = PIL.Image.Image.rotate(paperclip2,a,resample=PIL.Image.BICUBIC,expand=True)





        render_img.paste(paperclip2,(x,y),paperclip2)



    image_file = BytesIO()

    render_img.save(image_file, 'PNG')

    z.writestr(f'clips-{c+OFFSET}.png', image_file.getvalue())

    

df = pd.DataFrame( {'id':range(OFFSET,len(clip_count)+OFFSET),'clip_count':clip_count})

z.writestr('master.csv', df.to_csv(index=False))

z.close()

print("done")



df