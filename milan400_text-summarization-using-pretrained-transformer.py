import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install transformers
!pip install torch
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
text ="""
Nepal reported one more Covid-19 related death and 133 new infection cases on Friday. With this the national infection tally has reached 18,374 while the death toll has climbed to 44 The Health Ministry 
informed that an 85-year-old woman from Birgunj Metropolitan City Ward 11 in Parsa district died of Covid-19 at Narayani Hospital on Friday morning. “The deceased was a heart patient and she had been 
admitted to the hospital after suffering from pneumonia,” said Dr Jageshwor Gautam, the ministry spokesperson, at a press briefing. “Out of 3,987 real-time polymerase chain reaction tests carried out 
at 27 labs across the country, 133 samples came back positive,” said Gautam. Throat swabs of 31 people from Parsa, 25 from Jhapa, 10 from Kathmandu, eight each from Dailekh and Kailali; seven from 
Saptari, five each from Sunsari and Kaski; four each from Dang, Nawalparasi (East) and Rupandehi, three from Lamjung, two from Morang and once each from Dhankuta, Bara, Dhanusa, Udayapur, Baglung, 
Tanahun and Jajarkot tested positive. So far, 12,947 individuals have recovered from the disease. Gautam said 107 patients were discharged in the past 24 hours. As many as 335,082 PCR tests have been 
performed in the country so far. Six districts—Bhojpur, Panchthar, Sankhuwasabha, Rasuwa, Manang and Mustang—don’t have any active cases, Gautam said. On Thursday, Nepal had reported one Covid-19 related 
death and 147 new cases. Nepal had reported two Covid-19 related deaths and 100 new cases on Wednesday. There were 150 new cases on Tuesday and 186 new cases on Monday. On Sunday, the country reported 156
new cases while 57 new Covid-19 infections were detected on Saturday. The country reported its 40th Covid-19 related death and 101 new cases on Friday. Track all Covid-19 cases in Nepal here.
"""
preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)
tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)