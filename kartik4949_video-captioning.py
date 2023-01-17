import requests 
file_url = " http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip"
  
r = requests.get(file_url, stream = True) 
  
with open("flick8k.zip","wb") as pdf: 
    for chunk in r.iter_content(chunk_size=1024): 
         if chunk: 
             pdf.write(chunk) 

