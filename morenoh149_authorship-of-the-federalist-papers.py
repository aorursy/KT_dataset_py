import numpy as np
import pandas as pd
file = open('../input/The_federalist_papers.txt')
text = file.read()
text[:1000]
# TODO strip the author of each paper
# TODO cluster the papers to see if 3 clusters do show up
# TODO use DeepLearning to build generative model, https://gist.github.com/karpathy/d4dee566867f8291f086