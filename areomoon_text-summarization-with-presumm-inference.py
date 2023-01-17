!pip install -r /kaggle/input/pre-summy/PreSumm/requirements.txt
import pandas as pd 

summary_df = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')

summary_df
import pandas as pd

from nltk.tokenize import sent_tokenize



news_df = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')

news_txt = news_df['text']





with open('/kaggle/working/temp_news_text.txt', 'w') as save_txt:

    for i in range(50):

        save_txt.write(news_txt[i].strip() + '\n')
with open('/kaggle/working/temp_news_text.txt', 'r') as save_txt:

    f= save_txt.readlines()
print(f[0])
print(f[1])
!mkdir logs

!mkdir temp

!mkdir models

!mkdir result
import os

os.chdir('/kaggle/input/pre-summy/PreSumm/src/')
!python train.py -task abs -mode test_text -text_src /kaggle/working/temp_news_text.txt -test_from /kaggle/input/absbert-weights/model_step_148000.pt -model_path /kaggle/working/models -visible_gpus -1 -alpha 0.2 -result_path /kaggle/working/result/news -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/cnndm.log
import os

os.chdir('/kaggle/working/result/')
with open('news.-1.candidate', 'r') as save_txt:

    s= save_txt.readlines()

    
print(s[0])
print(s[1])
import os

os.chdir('/kaggle/working/')
!rm -r logs

!rm -r temp

!rm -r models

!rm -r result