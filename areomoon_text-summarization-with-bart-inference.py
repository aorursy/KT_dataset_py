# !pip install fairseq
# import pandas as pd

# df_summary = pd.read_csv("../input/news-summary/news_summary_more.csv")

# news_txt = df_summary['text']
# with open('/kaggle/working/news_text.txt','w') as save_txt:

#     for i in range(100):

#         save_txt.write(news_txt[i].strip()+'\n')
# import torch



# bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')

# bart.cuda()

# bart.eval()

# bart.half()

# count = 1

# bsz = 32
# with open('/kaggle/working/news_text.txt') as source, open('/kaggle/working/test.hypo', 'w') as fout:

#     sline = source.readline().strip()

#     slines = [sline]

#     for sline in source:

#         if count % bsz == 0:

#             with torch.no_grad():

#                 hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)



#             for hypothesis in hypotheses_batch:

#                 fout.write(hypothesis + '\n')

#                 fout.flush()

#             slines = []



#         slines.append(sline.strip())

#         count += 1
# with open('/kaggle/working/news_text.txt', 'r') as fout:

#     src=fout.readlines()
# with open('/kaggle/working/test.hypo', 'r') as fout:

#     s=fout.readlines()
# i=12

# print('Source text:'+'\n'+src[i])

# print('Summary text:'+'\n'+s[i])
