# When using the Kaggle consol the package download wouldn't persist. 

# OS.system being used instead.

import os

os.system("pip install git+https://github.com/user1342/Topic-Modelling-For-Online-Data.git")
from topic_modelling.topic_modelling import topic_modelling



# Return a list of the current topics and their keywords

modeller = topic_modelling()

print(modeller.get_topics())
# Visualise the topics

import pyLDAvis.gensim

pyLDAvis.display(modeller.get_lda_display())
# Identify the topic of a given news article

print(modeller.identify_topic("With the introduction of the General Data Protection Regulation (GDPR), the EU is enacting a set of mandatory regulations for businesses that go into effect soon, on May 25, 2018. Organisations found in non-compliance could face hefty penalties of up to 20 million euros, or 4 percent of worldwide annual turnover, whichever is higher."))
# Retrain the model. By default it uses a pre-provided list of 22545 news articles. However this can be changed. You can also specify the amount of passes and groups. 

modeller.re_train(number_of_topics=3, number_of_passes=15)

# Display the new model

pyLDAvis.display(modeller.get_lda_display())