from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '\\tesseract.exe'
import cv2
img=cv2.imread('\ocr.jpg')
print(pytesseract.image_to_string(img))
cv2.imshow('Result',img)
cv2.waitKey(0)