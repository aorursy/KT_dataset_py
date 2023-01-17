#驗證碼生成庫
from captcha. image import ImageCaptcha #pip install captcha
import numpy as np
from PIL import Image
import random
import sys
number = ['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number, captcha_size=4):
    #驗證碼列表
    captcha_text = []
    for i in range(captcha_size):
        #隨機選擇
        c = random.choice(char_set)
        #加入驗證碼列表
        captcha_text.append(c)
    return captcha_text

#生成字符對應的驗證碼
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    #獲得隨機生成的驗證碼
    captcha_text = random_captcha_text()
    #把驗證碼列表傳為字符串
    captcha_text = ''.join(captcha_text)
    #產生驗證碼
    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/' + captcha_text + '.jpg') #寫到文件

#數量少於10000， 因為重複名稱
num = 10000
if __name__ == '__main__' :
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    print("產生完畢")