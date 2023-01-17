from PIL import Image

import pytesseract

import os

import sys
!ls ../input/
class ImageQuoteExtractor():

    

    ### only single image will be feed here

    def load_single_image(self,imname):

        self.imname = imname

        self.im = Image.open(self.imname)

        ## convert to grayscale

        self.final = self.im.convert("L")

        return self.final

    ## in this a total folder will be feed here

    

    def translate_single_image(self):

        self.text = pytesseract.image_to_string(self.final) 

        return self.text

    def load_multiple_image(self,folder):

        self.images = []

        for filename in os.listdir(folder):

            ## load the image

            path = str(folder)+"/"+str(filename)

            #print(path)

            self.im = Image.open(path)

            self.tmp = self.im.convert('L')

            self.images.append(self.tmp)

    

    def translate_multiple_image(self):

        self.texts = []

        for image in self.images:

            self.text = pytesseract.image_to_string(image)

            self.texts.append(self.text)

        

        return self.texts
i = ImageQuoteExtractor()
i.load_single_image('../input/third.png')
i.translate_single_image()
i.load_multiple_image('../input')
i.translate_multiple_image()