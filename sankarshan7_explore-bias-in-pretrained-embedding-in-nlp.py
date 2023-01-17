import torch

from torch import nn

from tqdm import tqdm

from annoy import AnnoyIndex

import numpy as np

from typing import List, Set, Dict, Tuple, Optional
class PreTrainedEmbeddings(object):

    

    def __init__(self, 

                 word_to_index:Dict[str, int], 

                 word_vectors:List[np.ndarray]):

        

        self.word_to_index = word_to_index

        self.word_vectors = word_vectors

        self.index_to_words = {idx: word for word, idx in tqdm(self.word_to_index.items())}

        

        # Length of item vector that will be indexed

        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')

        print("Building Annoy Index...")

        for _, i in tqdm(self.word_to_index.items()):

            self.index.add_item(i, self.word_vectors[i])

            

        self.index.build(50) # 50 trees

        print("Finished!!")

        

    @classmethod

    def from_embedding_file(cls, embedding_file:str):

        """

        Instantiate from the embedding file

        

        Vector file should be of the format:

            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N

            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        

        Returns:

            Instance of the PretrainedEmbeddings

        """

        

        word_to_index = {}

        word_vectors = []

        

        print("Processing Embedding file ...")

        with open(embedding_file, 'r') as fp:

            for line in tqdm(fp.readlines()):

                

                line = line.split(" ")

                word = line[0]

                emb = np.array([float(x) for x in line[1:]])

                

                word_to_index[word] = len(word_to_index)

                word_vectors.append(emb)

                

        return cls(word_to_index, word_vectors)

    

    

    def get_embedding(self, word:str):

        """Given word, returns the embedding vector 

        

        Return:

            an embedding (np.ndarray)

        """

        return self.word_vectors[self.word_to_index[word]]

    

    def get_closest_to_vectors(self, vector:np.ndarray, n:int=1):

        """Given a vector, return its n nearest neighbours 

        

        Args: 

            vector (np.ndarray): should match the size of the vectors in the Annoy index

            n (int): the number of neighbours to return

            

        Returns:

            [str, str, str, ,...]: Words, those are nearest to the given word

                                   The words are not ordered by distance

                                   

        """

        nn_indices = self.index.get_nns_by_vector(vector, n)

        

        ls_word = [self.index_to_words[neighbours_idx] for neighbours_idx in nn_indices]

        return ls_word

    

    def compute_and_print_analogy(self, word1:str, word2:str, word3:str, k_nearest:int=1):

        

        emb1 = self.get_embedding(word1)

        emb2 = self.get_embedding(word2)

        emb3 = self.get_embedding(word3)

        

        # let's compute the 4th words mebedding!!

        # idea: w2 - w1 = w4 - w3

        spatial_relationship = emb2 - emb1

        emb4 = emb3 + spatial_relationship

        

        # get closest neighbours

        closest_words = self.get_closest_to_vectors(emb4, k_nearest)

        existing_words = set([word1, word2, word3])

        closest_words = [word for word in closest_words if word not in existing_words]

        

        if len(closest_words) == 0:

            print("Couldn't find closest words")

            print(f"{word1} : {word2} :: {word3} : ??")

        

        else:

            for w4 in closest_words:

                print(f"{word1} : {word2} :: {word3} : {w4}")

        
emb_file = "/kaggle/input/glove6b100dtxt/glove.6B.100d.txt"
embeddings = PreTrainedEmbeddings.from_embedding_file(emb_file)
embeddings.compute_and_print_analogy('man', 'he', 'woman', k_nearest=4)
embeddings.compute_and_print_analogy('fly', 'plane', 'sail', k_nearest=5)
embeddings.compute_and_print_analogy('leg', 'legs', 'hand', k_nearest=4)
embeddings.compute_and_print_analogy('happy', 'love', 'girl', k_nearest=4)
embeddings.compute_and_print_analogy('boy', 'girl', 'dream', k_nearest=4)
embeddings.compute_and_print_analogy('woman', 'doctor', 'man', k_nearest=4)