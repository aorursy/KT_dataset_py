from PIL import Image

import requests

from io import BytesIO



response = requests.get('http://anno.onb.ac.at/cgi-content/annoshow?call=bbr|19140728|1|33.0|0')

img = Image.open(BytesIO(response.content))
import pytesseract
!apt-get install tesseract-ocr-deu
img
print(pytesseract.image_to_string(img, lang='deu'))
img2 = Image.open('../input/bierbrauer-crop/bierbrauer_28_7_1914_preproc.jpg')
img2
print(pytesseract.image_to_string(img2, lang='deu'))