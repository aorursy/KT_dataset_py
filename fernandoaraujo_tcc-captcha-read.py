#https://www.profissionaisti.com.br/2010/07/experiencia-de-ocr-quebrando-captcha-com-26-linhas-de-codigo-python/
# Listando arquivos

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Realizando import da biblioteca que carrega imagem e analisa a mesma

from PIL import Image

import pytesseract

# Imagem a ser quebrada.

img = Image.open('/kaggle/input/zzcaptcha/captcha.jpg')

# Convertendo imagem para o padrão RGB

img = img.convert("RGBA")

# Realizando bind da imagem

pixdata = img.load()

# Convertendo imagem para apresentar fundo branco

for y in range(img.size[1]):

    for x in range(img.size[0]):

        if pixdata[x, y] != (0, 0, 0, 255):

            pixdata[x, y] = (255, 255, 255, 255) 

# Salvando nova imagem com fundo branco

img.save("captcha-white-background.gif", "GIF")

# Aumentando as dimensões da imagem (requerido pelo OCR)

im_orig = Image.open('captcha-white-background.gif')

big = im_orig.resize((116, 56), Image.NEAREST)

# Salvando imagem com tamanho maior

big.save("captcha-treated.tif")

image = Image.open('captcha-treated.tif')

# Apresentando imagem tratada

im_orig
# Imprimindo imagem em formato de string OCRizado

result = pytesseract.image_to_string(image)

result