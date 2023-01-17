import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.misc import imread
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
dataFile = pd.read_csv('../input/mbti-type/mbti_1.csv')
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

dataFile['words_per_comment'] = dataFile['posts'].apply(lambda x: len(x.split())/50)
dataFile['variance_of_word_counts'] = dataFile['posts'].apply(lambda x: var_row(x))

plt.figure(figsize=(15,10))
sns.swarmplot("type", "words_per_comment", data=dataFile)
#INTJ
dataFile_1 = dataFile[dataFile['type'] == 'INTJ']
text = str(dataFile_1['posts'].tolist())

INTJ_mask = np.array(Image.open('../input/mask-images/intj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=INTJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(INTJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('INTJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(INTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('ARCHITECT', loc='Center', fontsize=14)
plt.axis("off")
#INTP
dataFile_2 = dataFile[dataFile['type'] == 'INTP']
text = str(dataFile_2['posts'].tolist())

INTP_mask = np.array(Image.open('../input/mask-images/intp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=INTP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(INTP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('INTP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(INTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('LOGICIAN', loc='Center', fontsize=14)
plt.axis("off")
#ENTJ
dataFile_3 = dataFile[dataFile['type'] == 'ENTJ']
text = str(dataFile_3['posts'].tolist())

ENTJ_mask = np.array(Image.open('../input/mask-images/entj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ENTJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ENTJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ENTJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ENTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('COMMANDER', loc='Center', fontsize=14)
plt.axis("off")
#ENTP
dataFile_4 = dataFile[dataFile['type'] == 'ENTP']
text = str(dataFile_4['posts'].tolist())

ENTP_mask = np.array(Image.open('../input/mask-images/entp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ENTP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ENTP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ENTP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ENTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('DEBATER', loc='Center', fontsize=14)
plt.axis("off")
#INFJ
dataFile_5 = dataFile[dataFile['type'] == 'INFJ']
text = str(dataFile_5['posts'].tolist())

INFJ_mask = np.array(Image.open('../input/mask-images/infj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=INFJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(INFJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('INFJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(INFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('ADVOCATE', loc='Center', fontsize=14)
plt.axis("off")
#INFP
dataFile_6 = dataFile[dataFile['type'] == 'INFP']
text = str(dataFile_6['posts'].tolist())

INFP_mask = np.array(Image.open('../input/mask-images/infp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=INFP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(INFP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('INFP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(INFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('MEDIATOR', loc='Center', fontsize=14)
plt.axis("off")
#ENFJ
dataFile_7 = dataFile[dataFile['type'] == 'ENFJ']
text = str(dataFile_7['posts'].tolist())

ENFJ_mask = np.array(Image.open('../input/mask-images/enfj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ENFJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ENFJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ENFJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ENFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('PROTAGONIST', loc='Center', fontsize=14)
plt.axis("off")
#ENFP
dataFile_8 = dataFile[dataFile['type'] == 'ENFP']
text = str(dataFile_8['posts'].tolist())

ENFP_mask = np.array(Image.open('../input/mask-images/enfp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ENFP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ENFP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ENFP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ENFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('CAMPAIGNER', loc='Center', fontsize=14)
plt.axis("off")
#ISTJ
dataFile_9 = dataFile[dataFile['type'] == 'ISTJ']
text = str(dataFile_9['posts'].tolist())

ISTJ_mask = np.array(Image.open('../input/mask-images/istj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ISTJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ISTJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ISTJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ISTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('LOGISTICIAN', loc='Center', fontsize=14)
plt.axis("off")
#ISFJ
dataFile_10 = dataFile[dataFile['type'] == 'ISFJ']
text = str(dataFile_10['posts'].tolist())

ISFJ_mask = np.array(Image.open('../input/mask-images/isfj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ISFJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ISFJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ISFJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ISFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('DEFENDER', loc='Center', fontsize=14)
plt.axis("off")
#ESTJ
dataFile_11 = dataFile[dataFile['type'] == 'ESTJ']
text = str(dataFile_11['posts'].tolist())

ESTJ_mask = np.array(Image.open('../input/mask-images/estj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ESTJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ESTJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ESTJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ESTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('EXECUTIVE', loc='Center', fontsize=14)
plt.axis("off")
#ESFJ
dataFile_12 = dataFile[dataFile['type'] == 'ESFJ']
text = str(dataFile_12['posts'].tolist())

ESFJ_mask = np.array(Image.open('../input/mask-images/esfj.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ESFJ_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ESFJ_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ESFJ', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ESFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('CONSUL', loc='Center', fontsize=14)
plt.axis("off")
#ISTP
dataFile_13 = dataFile[dataFile['type'] == 'ISTP']
text = str(dataFile_13['posts'].tolist())

ISTP_mask = np.array(Image.open('../input/mask-images/istp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ISTP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ISTP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ISTP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ISTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('VIRTUOSO', loc='Center', fontsize=14)
plt.axis("off")
#ISFP
dataFile_14 = dataFile[dataFile['type'] == 'ISFP']
text = str(dataFile_14['posts'].tolist())

ISFP_mask = np.array(Image.open('../input/mask-images/isfp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ISFP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ISFP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ISFP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ISFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('ADVENTURER', loc='Center', fontsize=14)
plt.axis("off")
#ESTP
dataFile_15 = dataFile[dataFile['type'] == 'ESTP']
text = str(dataFile_15['posts'].tolist())

ESTP_mask = np.array(Image.open('../input/mask-images/estp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ESTP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ESTP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ESTP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ESTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('ENTREPRENEUR', loc='Center', fontsize=14)
plt.axis("off")
#ESFP
dataFile_16 = dataFile[dataFile['type'] == 'ESFP']
text = str(dataFile_16['posts'].tolist())

ESFP_mask = np.array(Image.open('../input/mask-images/esfp.png'))
stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=ESFP_mask,
               stopwords=stopwords,)
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(ESFP_mask)

# show
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.title('ESFP', loc='Center', fontsize=14)
plt.axis("off")

plt.subplot(122)
plt.imshow(ESFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.title('ENTERTAINER', loc='Center', fontsize=14)
plt.axis("off")