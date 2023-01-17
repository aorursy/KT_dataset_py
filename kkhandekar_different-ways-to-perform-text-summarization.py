article = "In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, 'With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow.' The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills."

print(article)
# - Libraries -

from nltk.corpus import stopwords

from nltk.cluster.util import cosine_distance

import numpy as np

import networkx as nx

import gc
# Custom Function: Split Sentences

def sentence_split(txt):

    prg = txt.split(". ")

    sentences = []

    

    for sentence in prg:

        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    sentences.pop()

    

    return sentences

    

# Custom Function: Check Sentence Similarity

def sentence_similarity(sent1, sent2, stopwords=None):

    if stopwords is None:

        stopwords = []

    

    sent1 = [w.lower for w in sent1]

    sent2 = [w.lower for w in sent2]

    

    all_words = list(set(sent1 + sent2))

    

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)    

    

    # Vector for the Sentence#1

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # Vector for the Sentence#2

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2)

    

# Custom Function: Similarity Matrix

def build_similarity_matrix(sentences, stop_words):

    # Empty Matrix

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

 

    for idx1 in range(len(sentences)):

        for idx2 in range(len(sentences)):

            if idx1 == idx2: #ignore if both are same sentences

                continue 

            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)



    return similarity_matrix

""" -- Combine all the above function into one -- """

# Custom Function: Generate Summary

def generate_summary(txt, top_n=5):

    

    stop_words = stopwords.words('english')

    summarize_text = []

    

    sentences = sentence_split(txt)  #Custom Function: Split Sentences

    

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)    #Custom Function: Check Sentence Similarity

    

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)     #Rank Sentences in Similarity Matrix

    scores = nx.pagerank(sentence_similarity_graph)

    

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)  #Sort Rank & pick top sentences

    #print("Indexes of top ranked_sentence order are ", ranked_sentence)     



    for i in range(top_n):

        summarize_text.append(" ".join(ranked_sentence[i][1]))



    print("Summarize Text: \n", ". ".join(summarize_text))       #Output Summarized Text    
# Generate Text Summary

generate_summary(article, 2)
# -Libraries -

from transformers import TFAutoModelWithLMHead, AutoTokenizer
model = TFAutoModelWithLMHead.from_pretrained("t5-base")

tokenizer = AutoTokenizer.from_pretrained("t5-base")
# Define Input

input = tokenizer.encode("summarize: " + article, return_tensors="tf", max_length=512)

output = model.generate(input, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,early_stopping=True)

print("Summarize Text: \n", tokenizer.decode(output[0], skip_special_tokens=True))
# -Library-

from transformers import pipeline
# Define Pipeline

summarizer = pipeline("summarization")
# Generate Summary

print(summarizer(article, max_length=150, min_length=40, do_sample=False))