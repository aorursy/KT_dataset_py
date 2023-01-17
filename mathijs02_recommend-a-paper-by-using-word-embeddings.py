!pip install spacy-langdetect
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import spacy



from gensim.models import Word2Vec

from sklearn import metrics

from sklearn import mixture

from sklearn import decomposition

from spacy_langdetect import LanguageDetector

from tqdm.auto import tqdm

from langdetect import DetectorFactory



pd.set_option('max_colwidth', 600)

tqdm.pandas()

DetectorFactory.seed = 42
df = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv").fillna("")
cord19_emb = Word2Vec.load('/kaggle/input/covid19-challenge-trained-w2v-model/covid.w2v')
def tokenize_columns(df, col_list):

    """

    Tokenize the string in multiple pandas dataframe columns.

    """

    def tokenize(text):

        """

        Tokenize one string, if the language is English.

        """

        exclusion_list = ["Unknown"]

        doc = nlp(text)



        if (doc._.language["language"] == "en") and (text not in exclusion_list):

            tokenized_text = [

                t.text.lower()

                for t in doc

                if not t.is_stop and t.is_alpha and len(t.text) > 1

            ]

            return tokenized_text

        else:

            return []



    nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)



    for col in col_list:

        df[col + "_tokenized"] = df[col].progress_apply(tokenize)

    return df





df = tokenize_columns(df, ["title", "abstract"])
def freq_absent(word_list, embedding, n_top=5):

    """

    Give the most frequent words that are missing in the embedding.

    """

    absent_list = [m for m in word_list if m not in embedding.wv]

    top_list = pd.Series(absent_list).value_counts()[:n_top]

    return top_list
freq_absent(df["title_tokenized"].dropna()[0:1000].sum(), cord19_emb)
freq_absent(df["abstract_tokenized"].dropna()[0:1000].sum(), cord19_emb)
def embed_columns(df, col_list, embedding):

    """

    Give one embedding vector per column (e.g. title or abstract).

    """

    def aggregate_embedding(item_list, embedding):

        """

        Calculate the embedding of each word and aggregate to a

        document-level embedding by taking the mean.

        """

        embedding_list = [embedding.wv[m] for m in item_list if m in embedding.wv]

        if len(embedding_list) > 0:

            return np.array(embedding_list).mean(axis=0)

        else:

            return None



    for col in col_list:

        df[col + "_embedding"] = df[col + "_tokenized"].apply(lambda x: aggregate_embedding(x, embedding))

    return df





df = embed_columns(df, ["title", "abstract"], cord19_emb)
class PaperClustering:

    

    def __init__(self, df, embedding_col, n_clusters):

        self.df = df.dropna(subset=[embedding_col])

        self.embedding_col = embedding_col

        self.n_clusters = n_clusters

    

    def plot_pca(self):

        """

        Do a PCA, keep the two components containing

        most variance and make a scatter plot.

        """

        pca_data = np.stack(self.df[self.embedding_col].to_numpy())

        pca = decomposition.PCA(n_components=2)

        transformed_data = pca.fit_transform(pca_data)

        self.df["pca_coord0"] = transformed_data.T[0]

        self.df["pca_coord1"] = transformed_data.T[1]



        if "color" in self.df.columns:

            c = self.df["color"]

        else:

            c = None



        plt.figure(figsize=(5, 5))

        plt.scatter(

            x=self.df["pca_coord0"],

            y=self.df["pca_coord1"],

            alpha=0.006,

            c=c

        )



    def cluster_data(self):

        """

        Cluster the data using a Gaussian mixture model.

        """

        cl = mixture.GaussianMixture(n_components=self.n_clusters, max_iter=500, random_state=10)

        fit_data = np.stack(self.df[self.embedding_col].values)

        self.df["cluster"] = cl.fit_predict(fit_data)

        self.df["color"] = self.df["cluster"].apply(lambda x: "C" + str(x))



    def cluster_size(self):

        """

        Return the cluster sizes.

        """

        return self.df["cluster"].value_counts(sort=False)



    def print_cluster_examples(self, n_top=10):

        """

        Print for each cluster n_top examples.

        """

        for n in range(self.n_clusters):

            subset = self.df[self.df["cluster"] == n]["title"].sample(n=n_top, random_state=42)



            print(f"\nCluster {n}\n---------")

            for m in subset:

                print(m)
paper_clusters = PaperClustering(df=df, embedding_col="abstract_embedding", n_clusters=3)
paper_clusters.plot_pca()
paper_clusters.cluster_data()

paper_clusters.plot_pca()
paper_clusters.cluster_size()
paper_clusters.print_cluster_examples()
ANCHOR_PAPER = 25



print(df.loc[ANCHOR_PAPER, "title"])
class SimilarPaper:

    

    def __init__(self, df, comparison_col):

        self.df = df.dropna(subset=[comparison_col])

        self.comparison_col = comparison_col



    def calculate_similarity(self, anchor_index):

        """

        Calculate the cosine distance of every embedding

        to a given 'anchor' embedding.

        """

        anchor_emb = self.df.loc[anchor_index, self.comparison_col]

        self.df["cos_sim"] = self.df[self.comparison_col].apply(

            lambda x: metrics.pairwise.cosine_similarity(

                [x],

                [anchor_emb]

            )[0][0]

        )



    def get_top_similar(self, n_top=10):

        """

        Return the n_top papers that are most similar

        to the anchor paper.

        """

        top_list = (

            self.df

            .sort_values("cos_sim", ascending=False)

            [["title", "cos_sim"]]

            .drop_duplicates()

            [:n_top]

        )

        return top_list
paper_recom = SimilarPaper(df=df, comparison_col="abstract_embedding")

paper_recom.calculate_similarity(anchor_index=ANCHOR_PAPER)

paper_recom.get_top_similar()