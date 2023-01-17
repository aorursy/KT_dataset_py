import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
raw_data = pd.read_csv('../input/train.csv').fillna('no comment')

test_data = pd.read_csv('../input/test.csv').fillna('no comment')
raw_data.info()
all_comments = pd.concat([raw_data.comment_text, test_data.comment_text])
sentences = [''.join(c for c in s if c not in string.punctuation) for s in all_comments]
sentences[:5]
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards','again', 'against', 'all', 'almost', 'alone', 'along','already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another','any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became','because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below','beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant','co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due','during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever','every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first','five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give','go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers','herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed','interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made','many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much','must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none','noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or','other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please','put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should','show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime','sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves','then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they','thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to','together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon','us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where','whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who','whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'yet', 'you', 'your','yours', 'yourself', 'yourselves']
stopwords += ['A', 'About', 'Above', 'Across', 'After', 'Afterwards','Again', 'Against', 'All', 'Almost', 'Alone', 'Along']
stopwords += ['Already', 'Also', 'Although', 'Always', 'Am', 'Among','Amongst', 'Amoungst', 'Amount', 'An', 'And', 'Another']
stopwords += ['Any', 'Anyhow', 'Anyone', 'Anything', 'Anyway', 'Anywhere','Are', 'Around', 'As', 'At', 'Back', 'Be', 'Became']
stopwords += ['Because', 'Become', 'Becomes', 'Becoming', 'Been','Before', 'Beforehand', 'Behind', 'Being', 'Below']
stopwords += ['Beside', 'Besides', 'Between', 'Beyond', 'Bill', 'Both','Bottom', 'But', 'By', 'Call', 'Can', 'Cannot', 'Cant']
stopwords += ['Co', 'Computer', 'Con', 'Could', 'Couldnt', 'Cry', 'De','Describe', 'Detail', 'Did', 'Do', 'Done', 'Down', 'Due']
stopwords += ['During', 'Each', 'Eg', 'Eight', 'Either', 'Eleven', 'Else','Elsewhere', 'Empty', 'Enough', 'Etc', 'Even', 'Ever']
stopwords += ['Every', 'Everyone', 'Everything', 'Everywhere', 'Except','Few', 'Fifteen', 'Fifty', 'Fill', 'Find', 'Fire', 'First']
stopwords += ['Five', 'For', 'Former', 'Formerly', 'Forty', 'Found','Four', 'From', 'Front', 'Full', 'Further', 'Get', 'Give']
stopwords += ['Go', 'Had', 'Has', 'Hasnt', 'Have', 'He', 'Hence', 'Her','Here', 'Hereafter', 'Hereby', 'Herein', 'Hereupon', 'Hers']
stopwords += ['Herself', 'Him', 'Himself', 'His', 'How', 'However','Hundred', 'I', 'Ie', 'If', 'In', 'Inc', 'Indeed']
stopwords += ['Interest', 'Into', 'Is', 'It', 'Its', 'Itself', 'Ieep','Last', 'Latter', 'Latterly', 'Least', 'Less', 'Ltd', 'Made']
stopwords += ['Many', 'May', 'Me', 'Meanwhile', 'Might', 'Mill', 'Mine','More', 'Moreover', 'Most', 'Mostly', 'Move', 'Much']
stopwords += ['Must', 'My', 'Myself', 'Name', 'Namely', 'Neither', 'Never','Nevertheless', 'Next', 'Nine', 'No', 'Nobody', 'None']
stopwords += ['Noone', 'Nor', 'Not', 'Nothing', 'Now', 'Nowhere', 'Of','Off', 'Often', 'On','Once', 'One', 'Only', 'Onto', 'Or']
stopwords += ['Other', 'Others', 'Otherwise', 'Our', 'Ours', 'Ourselves','Out', 'Over', 'Own', 'Part', 'Per', 'Perhaps', 'Please']
stopwords += ['Put', 'Rather', 'Re', 'S', 'Same', 'See', 'Seem', 'Seemed','Seeming', 'Seems', 'Serious', 'Several', 'She', 'Should']
stopwords += ['Show', 'Side', 'Since', 'Sincere', 'Six', 'Sixty', 'So','Some', 'Somehow', 'Someone', 'Something', 'Sometime']
stopwords += ['Sometimes', 'Somewhere', 'Still', 'Such', 'System', 'Take','Ten', 'Than', 'That', 'The', 'Their', 'Them', 'Themselves']
stopwords += ['Then', 'Thence', 'There', 'Thereafter', 'Thereby','Therefore', 'Therein', 'Thereupon', 'These', 'They']
stopwords += ['Thick', 'Thin', 'Third', 'This', 'Those', 'Though', 'Three','Three', 'Through', 'Throughout', 'Thru', 'Thus', 'To']
stopwords += ['Together', 'Too', 'Top', 'Toward', 'Towards', 'Twelve','Twenty', 'Two', 'Un', 'Under', 'Until', 'Up', 'Upon']
stopwords += ['Us', 'Very', 'Via', 'Was', 'We', 'Well', 'Were', 'What','Whatever', 'When', 'Whence', 'Whenever', 'Where']
stopwords += ['Whereafter', 'Whereas', 'Whereby', 'Wherein', 'Whereupon','Wherever', 'Whether', 'Which', 'While', 'Whither', 'Who']
stopwords += ['Whoever', 'Whole', 'Whom', 'Whose', 'Why', 'Will', 'With','Within', 'Without', 'Would', 'Yet', 'You', 'Your']
stopwords += ['Yours', 'Yourself', 'Yourselves']
frequency = nltk.FreqDist(sentences)
frequency.plot(25, cumulative = False)
frequency.most_common(5)
new_data_labels = raw_data.drop(['id', 'comment_text'], axis=1)
count = []
labels = list(new_data_labels.columns.values)
for i in labels:
    count.append((i, new_data_labels[i].sum()))
new_data_count = pd.DataFrame(count, columns=['label','comments'])
new_data_count
new_data_count.plot(x = 'label', y= 'comments', kind= 'bar', legend= False, grid= True)
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, use_idf=True,max_df=0.8,  min_df=1, ngram_range=(1,2))
vect_word = vect.fit(sentences)
train_vect = vect_word.transform(raw_data.comment_text)
test_vect = vect_word.transform(test_data.comment_text)
from sklearn.model_selection import train_test_split
train_data, dev_data, train_label, dev_label = train_test_split(train_vect,new_data_labels)
mlpclass =MLPClassifier(solver='lbfgs', alpha=1e-5, validation_fraction=0.3, hidden_layer_sizes=(4,4), verbose= True, activation= 'logistic', max_iter= 200 , learning_rate_init= 0.0001)
mlpclass.fit(train_data, train_label)
pred = mlpclass.predict(dev_data)
mlpclass.score(dev_data, dev_label)
test_pred = mlpclass.predict_proba(test_vect)
submission = pd.read_csv('../input/sample_submission.csv')
submission[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]=test_pred
submission.to_csv('submission.csv', index= False)

