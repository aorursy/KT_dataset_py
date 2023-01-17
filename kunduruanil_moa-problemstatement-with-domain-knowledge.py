

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import IPython



train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

target = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
# What is the Mechanism of Action (MoA) of a drug? And why is it important?



# In the past, scientists derived drugs from natural products or were inspired by traditional remedies.

# Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the 

# biological mechanisms driving their pharmacological activities were understood.  

# Today,with the advent of more powerful technologies, 

# drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the

# underlying biological mechanism of a disease. In this new framework, 

# scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target.

# As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action

# or MoA for short.

# 1. In the past, scientists derived drugs from natural products or were inspired by 

# """traditional remedies"""

IPython.display.Image("/kaggle/input/poaimages/main_-_arab_remedies.jpg")
# 2. Today,with the advent of more powerful technologies,

# more targeted model based on an understanding of the underlying biological mechanism of a disease.



# 3. scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target

IPython.display.YouTubeVideo('-5PlQcs5Pr8', width=1000, height=350)
# molecule that can modulate that protein target.

IPython.display.YouTubeVideo('u49k72rUdyc', width=1000, height=350)
# As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action

# or MoA for short.

IPython.display.YouTubeVideo('EEVGG0MdadE', width=1000, height=350)
## How do we determine the MoAs of a new drug?



# One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search 

# for similarity to known patterns in large genomic databases, 

# such as libraries of gene expression or cell viability patterns of drugs with known MoAs.



# I figured out that cell viability is the percentage of survived cells after treatment so it ranges either from 0 to 1 or from 0 to 100 

# (if in percent). Similarly, gene expression estimates 

# how well gene reacts with respect to overall genes so it's also a sort of ratio scaled as cells viability values

IPython.display.YouTubeVideo('OEWOZS_JTgk', width=1000, height=350)
# cell viability patterns of drugs

IPython.display.YouTubeVideo('wmR1p2M7VQ', width=1000, height=350)
target.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:20].plot(kind="barh",figsize=(20,10))
train.shape
target.shape