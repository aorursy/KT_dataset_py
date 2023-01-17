import pytesseract

from PIL import Image, ImageStat
__imp
PATH="data/myntra/test/"
files = os.listdir(f"{PATH}")


img = cv2.imread(img_path)

plt.imshow(img)

print(pytesseract.image_to_string(Image.open(img_path)))
def brightness( im_file ):

    im = Image.open(im_file).convert('L')

    stat = ImageStat.Stat(im)

    print("Read RMS brightness of image: ")

    print(str(im_file))

    print(stat.count)

    return stat.rms[0]
img_path = f"{PATH}826.jpg"

brightness(img_path)