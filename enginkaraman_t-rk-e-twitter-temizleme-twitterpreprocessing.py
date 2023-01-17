
import re   # tweet preprocessing için kullanılan en yaygın kütüphane 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def kucult(text):
    
    str      = text
    liste   = ''
    krktr = [('İ','i'), ('Ğ','ğ'),('Ü','ü'), ('Ş','ş'),
            ('Ö','ö'),('Ç','ç'), ('I','ı')]
    for liste, harf in krktr:
        str  = str.replace(liste, harf)
        str = str.lower()
        
    return str

def clean(text):
    
    text=re.sub(r'@[A-Za-z0-9]+','', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text=re.sub(r'https?:\/\/\S+','', text)
    text=re.sub(r'http?:\/\/\S+','', text)
    text=re.sub(r'RT[\s]+','', text)
    text=re.sub(r'\n','', text)
    text = re.sub(r"#(\w+)", ' ', text)
    text=re.sub(r'^\x00-\x7F]+','', text)
    text=re.sub(r'[^A-Za-zığüşöçİĞÜŞÖÇ]+',' ', text)
    text=re.sub(r'((https://[^\s]+))','', text)  
    text=kucult(text)
    
    return text
text='''RT @ophium52: Turizm Şirketleri İnsan saĞlığından önemli değildir. Mükemmmmel.  #SınavlarErtelensin https://t.co/tQg2QF0GFI

'''
text
text=clean(text)
text