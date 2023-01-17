import numpy as np
from PIL import Image
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import wordcloud
import nltk # for removing stopwords from word set
from nltk.corpus import stopwords

# if you have problems with using stopwords.words("english")
# you should manually download stopword dictionart by running the code below
# nltk.download("stopwords")
ESW = stopwords.words("english")
ESW = ESW + ["—", "would", "didn’t", "don’t", "it."]
def read_docu(document):
    
    words = []
    
    with open(document, "r", encoding = "utf-8") as input_file:
        for line in input_file:
            line = line.lower()
            line = line.strip().split()
            words += line
            
        return words
def stopword_filter(words):
    
    filtered_words = []
    
    for word in words:
        if not word in ESW:
            filtered_words.append(word)
        
    return filtered_words
def word_counter(words):
    
    word_counts = Counter(words)
        
    return word_counts
def bgimg_get(imgfile):
    
    bgimg = np.array(Image.open(imgfile))
    
    return bgimg
def set_wordcloud(word_counts, bgimg):
    
    cloud = wordcloud.WordCloud(background_color="white", mask = bgimg, collocations = False)
    word_cloud = cloud.generate_from_frequencies(word_counts)
    
    return word_cloud
def draw_wordcloud(document, imgfile):
    
    words = read_docu(document)
    fd_words = stopword_filter(words)
    word_counts = word_counter(fd_words)
    bgimg = bgimg_get(imgfile)
    word_cloud = set_wordcloud(word_counts, bgimg)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()
jobs_speech = "../input/jobscommencement/jobscommencement.txt"
jobs_img = "../input/stevejobs-pic/stevejobs.jpg"
draw_wordcloud(jobs_speech, jobs_img)
import squarify
def draw_treemap(document):
    
    # read document and count words
    words = read_docu(document)
    fd_words = stopword_filter(words)
    word_counts = word_counter(fd_words)
    
    # settings for treemap
    common_n = [n for (w, n) in word_counts.most_common(30)] # top30 words' counts
    common_w = [w for (w, n) in word_counts.most_common(30)] # top30 words
    norm = matplotlib.colors.Normalize(vmin = min(common_n), vmax = max(common_n))
    # you can change color(set with Blues now) here
    colors = [matplotlib.cm.Blues(norm(value)) for value in common_n] 
    
    # draw a treemap
    plt.figure(figsize=(10, 10))
    squarify.plot(label = common_w, sizes = common_n, color = colors, alpha = 0.5)
    plt.title("Word Treemap")
    plt.axis("off")
    plt.show()
draw_treemap("../input/jobscommencement/jobscommencement.txt")