%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from scipy.sparse import csr_matrix



import seaborn as sns



%config InlineBackend.figure_format = 'retina'
tags = pd.read_csv("../input/Tags.csv")
tags.head()
tags.shape
tags = tags.dropna() #there are a few NA tags - let's drop these.
(tags["Tag"].value_counts()

            .head(10)

            .plot(kind = "barh"))
(tags.groupby("Id")["Tag"]

     .count()

     .value_counts()

     .plot(kind = "bar"))
popular_tags = tags.Tag.value_counts().iloc[:1000].index
tag_counts = tags.groupby("Id")["Tag"].count()

many_tags = tag_counts[tag_counts >3].index



tags = tags[tags["Id"].isin(many_tags)] #getting questions with 5 tags

tags = tags[tags["Tag"].isin(popular_tags)] #getting only top 1000 tags
tags.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

from sklearn.pipeline import make_pipeline
#let's integer encode the id's and tags:

tag_encoder = LabelEncoder()

question_encoder = LabelEncoder()



tags["Tag"] = tag_encoder.fit_transform(tags["Tag"])

tags["Id"] = question_encoder.fit_transform(tags["Id"])
tags.head()
X = csr_matrix((np.ones(tags.shape[0]), (tags.Id, tags.Tag)))
X.shape #one row for each question, one column for each tag
model = TruncatedSVD(n_components=3)
model.fit(X)
two_components = pd.DataFrame(model.transform(X), columns=["one", "two", "three"])
two_components.plot(x = "one", y = "two", kind = "scatter", title = "2D PCA projection components 1 and 2")
two_components.plot(x = "two", y = "three", kind = "scatter", title = "2D PCA projection - components 2 and 3")
tagz = popular_tags[:20]



tag_ids = tag_encoder.transform(tagz)

n = len(tag_ids)
X_new = csr_matrix((np.ones(n), (pd.Series(range(n)), tag_ids)), shape = (n, 998))
proj = pd.DataFrame(model.transform(X_new)[:,:2], index=tagz, columns = ["one", "two"])

proj["tag"] = proj.index
from ggplot import * #ggplot!
plt = (ggplot(proj, aes(x = "one", y = "two", label = "tag")) +

            geom_point() +

            geom_text())

plt.show()
sm_proj = proj[proj["one"] < 0.2][proj["two"] < 0.2]

plt = (ggplot(sm_proj, aes(x = "one", y = "two", label = "tag")) +

            geom_point() +

            geom_text() +

            xlim(0, 0.1) +

            ylim(-0.1, 0.2))
plt.show()