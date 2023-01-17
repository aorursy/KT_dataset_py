from collections import defaultdict

import pandas as pd



def load(corpus):

    df = pd.read_csv('../input/Papers.csv')

    keys = df.columns



    for i in range(df.index.size):

        paper = Paper()

        for key, val in zip(keys, df.iloc[i]):

            setattr(paper, key, val)

        corpus.list.append(paper)



    names = pd.read_csv('../input/Authors.csv')

    names = dict(zip(names.Id, names.Name))



    paper_name = pd.read_csv('../input/PaperAuthors.csv')



    idx = defaultdict(list)



    for i in range(paper_name.index.size):

        _, PaperId, AuthorId = list(paper_name.iloc[i])

        idx[PaperId].append(names[AuthorId])



    for paper in corpus:

        paper.Authors = idx[paper.Id]



class Paper:

    def __init__(self):

        '''

        Attributes:

        Id, Title, Authors, EventType, PdfName, Abstract, PaperText

        '''

        pass



class Corpus:

    def __init__(self):

        self.list = []

        load(self)



    def __getitem__(self, idx):

        return self.list[idx]



    def __iter__(self):

        return iter(self.list)
#  Examples:

corpus = Corpus()

paper = corpus[10]

print(paper.Title, '\n', paper.Authors)
for paper in corpus:

    if 'posterior' in paper.Abstract.lower():

        print(paper.Title)