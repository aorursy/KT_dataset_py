# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from random import randint

import numpy as np # linear algebra

import cv2



import matplotlib.pyplot as plt_

plt_.rcParams["figure.figsize"] = (17,15)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for filenames in os.listdir('/kaggle/input'):

    print(filenames)



work_dir = '/kaggle/input/'



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def _deserialize_rects(file):

    "Parses data about true crop rects from the file"

    rects = {}

    for row in file.readlines():

        rect_data = row.strip().split()

        rects[rect_data[0]] = Rect(map(int, rect_data[1:]), is_corners=True)

    file.close()

    return rects
class Rect:



    def __init__(self, x, y=None, width=None, height=None, is_corners=False):

        """

            x - x1 coordinate x for top left corner

            y - y1 coordinate y for top left corner

            width - width of the rect if is_coreners is False (default), x2 - x of bottom right corner otherwise

            height - height of the rect if is_coreners is False (default), y2 - y of bottom right corner otherwise

        """

        try:

            x, y, width, height = x

        except TypeError:

            if None in (x, y, width, height):

                raise AttributeError

        if is_corners:

            x2, y2 = width, height

            if x > x2:

                self.x1, self.x2 = x2, x

            self.x1, self.x2 = x, x2

            if y > y2:

                self.y1, self.y2 = y2, y

            self.y1, self.y2 = y, y2

            return

        else:

            if width>0 and height>0:

                self.x1, self.x2 = x, x + width

                self.y1, self.y2 = y, y + height

                return

        raise AttributeError



    @property

    def point1(self) -> tuple:

        "(x, y) top left corner"

        return self.x1, self.y1



    @property

    def point2(self) -> tuple:

        "(x, y) bottom left corner"

        return self.x2, self.y2



    @property

    def width(self):

        "width of the rect"

        return self.x2 - self.x1



    @property

    def height(self):

        "height of the rect"

        return self.y2 - self.y1



    @property

    def S(self):

        "Rect area"

        return (self.x2 - self.x1) * (self.y2 - self.y1)



    def __or__(self, other):

        "Enclosing rect"

        return Rect(min(self.x1, other.x1), min(self.y1, other.y1),

                    max(self.x2, other.x2), max(self.y2, other.y2), is_corners=True)



    def rect_distance(self, other):

        "Distance between rects. 0 if rects intersect or touching"

        point1, point2 = np.array(self.point1), np.array(self.point2)

        point1b, point2b = np.array(other.point1), np.array(other.point2)

        self_extent, other_extent = (point2 - point1) / 2, (point2b - point1b) / 2

        self_center, other_center = self_extent + point1, other_extent + point1b

        xy = np.abs(self_center - other_center) - (self_extent + other_extent)

        xy[xy < 0] = 0

        xy[0] *= 2

        return (xy**2).sum()



    def draw(self, image, color=(255, 30, 255)):

        "drow rect on image with color. If color is 'r' generates random one"

        if color == "r":

            color = (randint(10, 255), randint(10, 255), randint(10, 255))

        cv2.rectangle(image, self.point1, self.point2, color, 5)



    def __add__(self, other):

        "Move rect on x and y values specified in other"

        return Rect(self.x1 + other[0], self.y1 + other[1],

                    self.x2 + other[0], self.y2 + other[1], is_corners=True)



    def __ne__(self, other):

        "Equality of the rects with 5% error"

        thres_X = self.width * 0.05

        thres_Y = self.height * 0.05

        return (abs(self.x1 - other.x1) > thres_X or

                abs(self.x2 - other.x2) > thres_X or

                abs(self.y1 - other.y1) > thres_Y or

                abs(self.y2 - other.y2) > thres_Y)



    def __eq__(self, other):

        return not (self != other)
def is_good_contour(contour):

    "Filter found conours. Ignore small noises and hard elongated"

    rect = cv2.boundingRect(contour)

    x, y, w, h = rect

    if 0.15 < w/h < 22:

        if w > 100 and h > 10 and w*h > 3000:

            return rect

    return False
os.mkdir("steps")

class Plt:

    counter = 0

    def imshow(self, image, *args, **kwargs):

        plt_.imsave(f"steps/{self.counter}.png", image, **kwargs)

        Plt.counter += 1

        print("writed", Plt.counter)

        return plt_.imshow(image, *args, **kwargs)

    def show(self):

        return plt_.show()

plt = Plt()
def content_area(image, intermediate_states = False):

    image1 = image

    if intermediate_states:

        plt.imshow(image)

        plt.show()

    grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # greyscaled image to work with

    if intermediate_states:

        plt.imshow(~grey, cmap="Greys")

        plt.show()

    blur = cv2.GaussianBlur(grey, (3, 3), 0) # blur image to hide noise

    if intermediate_states:

        plt.imshow(~blur, cmap="Greys")

        plt.show()

    edged = cv2.Canny(blur, 0, 20)

    if intermediate_states:

        plt.imshow(edged)

        plt.show()

    edged = ~cv2.dilate(edged, np.ones((3, 3))) # Detect edeges and dilate them to connect little breakes

    if intermediate_states:

        plt.imshow(edged)

        plt.show()

    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(edged, connectivity=4)

    # find background as largest connected component. Most often backgraund represents as solid monocolor area

    stats = stats[1:]

    label = np.argmax(stats[:, -1]) # find the largest one. Maybe it is not the best idea

    frame = stats[label] # Bounding box of the found component

    component = (regions != label+1)[frame[1]+2:frame[1]+frame[3]-2, frame[0]+2:frame[0]+frame[2]-2].astype(np.uint8)

    

    if intermediate_states:

        plt.imshow(component)

        plt.show()

    # Crop to the background component and binarize result as background 0 and others 1

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,

                                   cv2.CHAIN_APPROX_SIMPLE)

    # find contours in background

    offset = frame[[0, 1]] + 2 # store cropping offset

    if len(contours) < 20: # if its more then 20 countours foind, so most likely there's no easy detect content block or no background at all

        rects = [Rect(cv2.boundingRect(cont)) for cont in contours] # list of bounding rects of contours

        if intermediate_states:

            image1 = image.copy()

            for rect in rects:

                (rect + offset).draw(image1)

            plt.imshow(image1)

            plt.show()

        max_cnt = max(rects, key=lambda x: x.S) # largest contour area

        s = sum(rect.S for rect in rects) # Area of all contours

        if max_cnt.S/s > 0.95: # if there is large contour relative to others, so its most likely content block

            return max_cnt + offset

    

    component = cv2.dilate(component, np.ones((5, 5), np.uint8)) # dilate to union very close contours (like separate letters into text block)

    

    if intermediate_states:

        plt.imshow(component)

        plt.show()



    grey = grey[frame[1]+2:frame[1]+frame[3]-2, frame[0]+2:frame[0]+frame[2]-2]

    if abs(grey[component == 0].mean() - 255) < 2:# if the largest component is white, so we need to around all entire content

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,

                                       cv2.CHAIN_APPROX_SIMPLE)

        good_contours = []

        rects = []

        for cont in contours:

            brect = Rect(cv2.boundingRect(cont))

            

            if brect.S > 75000 or (brect.x1 > 50 and brect.x2 < component.shape[1] - 50 and 

                brect.y1 > 50 and brect.y2 < component.shape[0] - 50 and brect.S > 500):

                good_contours.append(cont)

                rects.append(brect)

        if intermediate_states:

            image1 = image.copy()

            for rect in rects:

                (rect + offset).draw(image1)

            plt.imshow(image1)

            plt.show()

        return Rect(cv2.boundingRect(np.concatenate(good_contours))) + offset

    

    # Otherwise it is no eye-catching block with content. So build it with paths of founded content

    edged = ~edged[frame[1]+2:frame[1]+frame[3]-2, frame[0]+2:frame[0]+frame[2]-2]

    if intermediate_states:

        plt.imshow(edged)

        plt.show()

    dilated = cv2.dilate(edged, np.ones((5, 5)))

    

    if intermediate_states:

        plt.imshow(dilated)

        plt.show()

    contours, _ = cv2.findContours(component, cv2.RETR_LIST, # findes contours on edges with enternal ones

                                   cv2.CHAIN_APPROX_SIMPLE)



    rects = []

    for contour in contours:

        rect = is_good_contour(contour)

        if rect and rect[2] < image1.shape[1]-2 and rect[3] < image1.shape[0]-2:

            rects.append(Rect(rect)) # filter countours with size of an hole image. we need enternal only

    rects.sort(key=lambda x: x.S, reverse=True) # sorts by size

    

    if intermediate_states:

        image1 = image.copy()

        for rect in rects:

            rect += offset

            rect.draw(image1, color="r")

        plt.imshow(image1)

        plt.show()

    

    #Iteratively union close rects

    is_change = True

    while is_change:

        rects_new = []

        is_change = False

        for i, rectl in enumerate(rects):

            j = i+1

            while j < len(rects):

                rectr = rects[j]

                dist = rectl.rect_distance(rectr)

                if dist < 1000:

                    is_change = True

                    rectl = rectl | rects.pop(j)

                else:

                    j += 1

            rects_new.append(rectl)

        rects = rects_new

    rects = rects[:2] # take two largest rects

    if len(rects) == 2: # if it's at least two rects

        if rects[1].S / rects[0].S > 2.2/3:  # if this rects have comparable sizes

            rect = rects[0] | rects[1] # so union them. It most likely founded two pages

        elif rects[1].S / rects[0].S >= 1/2: # exclusively for few readers that has incomlete (previous and next) pages on the screen

            if rects[0].x1 < 70 or rects[0].x2 > frame[2]-70: # and first or last page is smaller then incomplte book spred

                rect = rects[1]

            else:

                rect = rects[0]

        else:

            rect = rects[0]

    else:

        rect = rects[0]



    return rect + offset
image = cv2.cvtColor(cv2.imread(f"/kaggle/input/data/72.png"), cv2.COLOR_BGR2RGB)

rect = content_area(image, intermediate_states=True)

rect.draw(image)

plt.imshow(image)

plt.show()
image = cv2.cvtColor(cv2.imread(f"/kaggle/input/data/14.png"), cv2.COLOR_BGR2RGB)

rect = content_area(image, intermediate_states=True)

rect.draw(image)

plt.imshow(image)

plt.show()
def check_folder(folder='data/'):

    import shutil

    from datetime import datetime

    from time import time

    

    folder = work_dir + folder

    

    true_frames = open(work_dir + "frames.txt", "r")

    true_frames = _deserialize_rects(true_frames)



    good = 0

    bad = 0

    

    for dir_ in ("good", "bad"):

        if os.path.exists(dir_) and os.path.isdir(dir_):

            shutil.rmtree(dir_)

        os.mkdir(dir_)

    #clean_folder("results")

    #clean_folder("problems")



    log = open("log.log", "a")

    log.write("***************************************************************************************\n")

    log.write(str(datetime.now()) + '\n')

    

    t0 = time()

    for file_name in os.listdir(folder):

        image = cv2.imread(f"{folder}{file_name}")

        try:

            rect = content_area(image)

            rect.draw(image)

            if true_frames[file_name] == rect:

                cv2.imwrite(f"good/{file_name}", image)

                good += 1

            else:

                cv2.imwrite(f"bad/{file_name}", image)

                bad += 1

        except Exception as e:

            bad += 1

            cv2.imwrite(f"bad/{file_name}", image)

            print(f"Exception on file {file_name}")

            log.write("\nException: {}".format(type(e).__name__) + "\nException message: {}\n".format(e))

    log.write(f"time is {time()-t0}s")

    log.write(f"good={good} bad={bad} accuracy {good / (good + bad)}\n")

    log.close()
if __name__ == "__main__":

    pass

    #check_folder()
# with open("log.log") as log:

#     print(log.read())
# from random import sample

# good_files = os.listdir("good")

# bad_files = os.listdir("bad")

# for dir_, files in (("good/", good_files), ("bad/", bad_files)):

#     for file in sample(files, 2):

#         plt.imshow(cv2.cvtColor(cv2.imread(dir_ + file),cv2.COLOR_BGR2RGB))

#         plt.show()
def test_folder(folder1='test/'):

    import shutil

    from datetime import datetime

    from time import time

    

    folder = work_dir + folder1



    dir_ = f"{folder1.replace('/', '')}_results"

    if os.path.exists(dir_) and os.path.isdir(dir_):

        shutil.rmtree(dir_)

    os.mkdir(dir_)



    log = open("log.log", "a")

    log.write("***************************************************************************************\n")

    log.write(str(datetime.now()) + '\n')

    

    t0 = time()

    for file_name in os.listdir(folder):

        image = cv2.imread(f"{folder}{file_name}")

        path = dir_ + "/" + file_name

        try:

            rect = content_area(image)

            rect.draw(image)

        except Exception as e:

            print(f"Exception on file {file_name}")

            log.write("\nException: {}".format(type(e).__name__) + "\nException message: {}\n".format(e))

        cv2.imwrite(path, image)

    log.write(f"time is {time()-t0}s")

    log.close()
#test_folder()
# with open("log.log") as log:

#     print(log.read())
#test_res_files = os.listdir("test_results")

#for file in sample(test_res_files, 10):

#    plt.imshow(cv2.cvtColor(cv2.imread("test_results/" + file),cv2.COLOR_BGR2RGB))

#    plt.show()