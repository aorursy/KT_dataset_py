!pip install tensorflow==2.1
!pip install transformers
from transformers import pipeline
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf") # T5 (Text-To-Text Transfer Transformer) 's summarizer function

summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=10)
# directory = 'C:/Users/N1110/Desktop/CORD19/march30KeywordSearchQ1/'
# df = pd.read_csv(directory + 'mostRelatedQ1POLYMERASEabstract.csv')  #20 paper abstracts of size 300 on average
abs1="ltpgtltbgtltbgt there is no specific drug that has been approved for 2019ncov there are a number of factors that pose major challenges in their development approaches to the development of anti2019ncov include screening existing broadspectrum antiviral drugs repositioning of readily available clinical compounds and ltitalicgtde novoltitalicgt development of novel and specific agents for 2019ncov candidate compounds can be developed either to inhibit virusbased targets such as rna proteases polymerase spike glycoproteins and viral envelop and membrane proteins or to inhibit hostbased targets such as receptors and proteases that are utilized by virus for viral entry and endocytosis recently the rna polymerase remdesivir had demonstrated clinical efficacy in one patient with severe novel coronavirus pneumonia ncp the broadspectrum viral protease inhibitor kaletraltsupgtregltsupgt is also recommended in the current ncp clinical practice both drugs had lately been proceeded into multiple controlled phase iii clinical trials to test their safety and efficacy in ncp combinational therapies consisting of multiple drugs provide other viable options against 2019ncov based on scientific and clinical rationales using bioinformatics and database analysis we have identified 75 clinically compounds including 20 marketed compounds that are efficacious in inhibiting key targets in virus and hostbased approaches which may facilitate the development of new therapeutic options for 2019ncov ltpgt"
summarizer(abs1, min_length=50, max_length=100) # 300 words summary run for so long?
# original 208 words; summary 67 words


summarizer(abs1, min_length=20, max_length=50)
#even short summary 37 words and still valid
# so start to apply summarizer on the Question Answering Outputs (20 paper abstracts of size 300 on average)
import numpy as np
import pandas as pd
data_path = "/content/drive/My Drive/Colab Notebooks/cord19/data/"
data_filename = data_path +"mostRelatedQ1POLYMERASEabstract.csv"
print(data_filename)
# allow access to google drive files
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv(data_filename)
df.head(5)
df.shape[0]
df.ABS[0]
abs=[]


for i in range(df.shape[0]):
  abs.append(df.ABS[i])
  print("Summary for " + str(i)+"th Abstract is:")
  print(summarizer(abs[i], min_length=20, max_length=50))
  
for i in range(df.shape[0]):
  abs.append(df.ABS[i])
  print("Summary for " + str(i)+"th Abstract is:")
  #print(summarizer(abs[i], min_length=20, max_length=50))
  df["index"].iloc[i]= str(summarizer(abs[i], min_length=20, max_length=50)[0].values()) #df["index"].iloc[0]= str(summary0[0].values())
  print(df["index"].iloc[i]) #index should be named "summary"
type(summarizer(abs1, min_length=20, max_length=50))
summary0= summarizer(abs[0], min_length=20, max_length=50)


# summary0.values()

summary0
summary0[0].values()
df["index"].iloc[0]= str(summary0[0].values())

df["index"]
ABSsummary = ' '.join(df["index"])
ABSsummary
# put all the summaries of 20 paper abstracts together to generate a 600-1000 words summary

# in this way we let T5 help us read the 20 abstracts output
# further generate word cloud for ABSsummary to viz info