import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from wordcloud import WordCloud ,STOPWORDS



print("WordCloud is Imported")
alice_novel = open('../input/alice-novel1/alice_novel.txt', 'r').read()    
stopwords = set(STOPWORDS)



#We use stopwords that we imported from word_cloud. We use the function set to remove any redundant stopwords.
alice_wc = WordCloud(

    background_color='white',

    max_words=2000,

    stopwords=stopwords

)   

#generate word cloud from first 2000 words only
alice_wc.generate(alice_novel)  #generate the wordcloud
plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()          #display word cloud
fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



# display the cloud

plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
stopwords.add('said') # add the words said to stopwords



# re-generate the word cloud

alice_wc.generate(alice_novel)



# display the cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
# save mask to alice_mask

alice_mask = np.array(Image.open('../input/alice-mask1/alice_mask.png'))

    

print('Image downloaded and saved!')
#to preview mask

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')

plt.axis('off')

plt.show()
#Shaping the word cloud according to the mask is straightforward using word_cloud package. 

#For simplicity, we will continue using the first 2000 words in the novel.



# instantiate a word cloud object

alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)



# generate the word cloud

alice_wc.generate(alice_novel)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()