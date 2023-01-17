
%matplotlib inline
import os
import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sl
text_data = pd.read_csv("../input/facebook-antivaccination-dataset/posts_full.csv")
list(text_data.columns)
text_data.article_host.value_counts().nlargest(15).plot(kind='bar',
                                                       figsize=(15,5),
                                                       alpha=0.9,
                                                        rot=35,
                                                       title="Top 15 AntiVaxx Content Posting Sources")
text_data.anti_vax.value_counts().plot(kind='bar',
                                      rot=35,
                                      title="Count of Vax AntiVax Posts")