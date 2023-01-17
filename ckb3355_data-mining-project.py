from OCRengine import OCR
import os

path2 = "C:\\Users\\DELL\\OneDrive\\Desktop\\New folder\\Raw Images"
directory = os.path.join(path2)
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            inputPath = os.path.join(path2, file)
            input = OCR(file)
            input.extraction()