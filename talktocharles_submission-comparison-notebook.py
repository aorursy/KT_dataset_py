import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_colwidth = 500

sub1 = pd.read_csv('../input/nb-svm-strong-linear-baseline/submission.csv')
sub2 = pd.read_csv('../input/improved-lstm-baseline-glove-dropout/submission.csv')

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
res_compare = sub1.copy()
res_compare[label_cols] = (sub1[label_cols] - sub2[label_cols])**2
res_compare['diff'] = res_compare.sum(axis=1)
res_compare['comment_text'] = test['comment_text']
res_compare.sort_values('diff', ascending=False).loc[:,['id','comment_text','diff']]
sub1[sub1.id=='76cb5742586f4c2e'] #NB-SVM
sub2[sub2.id=='76cb5742586f4c2e'] #LSTM
test[test.id=='76cb5742586f4c2e'] #reminder of the comment
res_compare.to_csv('comparison.csv')
