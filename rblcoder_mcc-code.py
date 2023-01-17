import pandas as pd
!java -version
!pip install -q tabula-py



import tabula



tabula.environment_info()



import tabula

pdf_path = "../input/mcc-code-citibank/Merchant-Category-Codes.pdf"



dffrompdf = tabula.read_pdf(pdf_path, stream=True)
print(len(dffrompdf))

dffrompdf[0]

dffrompdf[0].info()
dffrompdf[1].info()
dfs = tabula.read_pdf('../input/mcc-code-citibank/Merchant-Category-Codes.pdf', pages="all")

len(dfs)
df_allpages = pd.concat(dfs)
df_allpages.info()
df_allpages.head()
df_allpages.tail()
!git clone https://github.com/greggles/mcc-codes.git
!ls mcc-codes


df = pd.read_csv('mcc-codes/mcc_codes.csv')
df.info()
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline



text_clf = Pipeline([

     ('vect', TfidfVectorizer()),

     ('clf', SGDClassifier(loss='hinge', penalty='l2',

                           alpha=1e-3, random_state=42,

                           max_iter=5, tol=None)),

 ])
text_clf.fit(df['edited_description'], df['mcc'])
predicted = text_clf.predict(df['edited_description'])
from sklearn import metrics

print(metrics.classification_report(df['mcc'], predicted))
from sklearn.metrics import jaccard_score

jaccard_score(df['mcc'], predicted, average='weighted')
from sklearn.metrics import f1_score

f1_score(df['mcc'], predicted, average='weighted')
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(df['mcc'], predicted)