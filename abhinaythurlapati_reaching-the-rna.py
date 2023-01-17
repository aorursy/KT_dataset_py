# Import packages and modules 



import pandas as pd

import numpy as np

import os

from os import path

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns





# Initialize the paths of the dataset



dataset_dir_path = '/kaggle/input/stanford-covid-vaccine'



bpps_path = path.join(dataset_dir_path, 'bpps')

train_data_path = path.join(dataset_dir_path, 'train.json')

test_data_path = path.join(dataset_dir_path, 'test.json')

sample_submission_path = path.join(dataset_dir_path, 'sample_submission.csv')
class Model:

    @staticmethod

    def gru_layer(hidden_dim, dropout):

        return L.Bidirectional(L.GRU(

            hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))



    @staticmethod

    def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5,

                    sp_dropout=0.2, embed_dim=200, hidden_dim=256, n_layers=3):

        inputs = L.Input(shape=(seq_len, 2))

        embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)



        reshaped = tf.reshape(

            embed, shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3])

        )

        hidden = L.SpatialDropout1D(sp_dropout)(reshaped)



        for x in range(n_layers):

            hidden = Model.gru_layer(hidden_dim, dropout)(hidden)



        # Since we are only making predictions on the first part of each sequence,

        # we have to truncate it

        truncated = hidden[:, :pred_len]

        out = L.Dense(5, activation='linear')(truncated)



        model = tf.keras.Model(inputs=inputs, outputs=out)

        model.compile(tf.optimizers.Adam(), loss=HelperFunctions.MCRMSE)



        return model





class Columns:

    index = 'index'

    id = 'id'

    sequence = 'sequence'

    seq_length = 'seq_length'

    seq_scored = 'seq_scored'

    structure = 'structure'

    signal_to_noise = 'signal_to_noise'

    SN_filter = 'SN_filter'

    predicted_loop_type = 'predicted_loop_type'

    reactivity_error = 'reactivity_error'

    deg_error_Mg_pH10 = 'deg_error_Mg_pH10'

    deg_error_pH10 = 'deg_error_pH10'

    deg_error_Mg_50C = 'deg_error_Mg_50C'

    deg_error_50C = 'deg_error_50C'

    reactivity = 'reactivity'

    deg_Mg_pH10 = 'deg_Mg_pH10'

    deg_pH10 = 'deg_pH10'

    deg_Mg_50C = 'deg_Mg_50C'

    deg_50C = 'deg_50C'





class TestColumns:

    index = 'index'

    id = 'id'

    sequence = 'sequence'

    seq_length = 'SN_filter'

    seq_scored = 'seq_scored'

    structure = 'structure'

    predicted_loop_type = 'predicted_loop_type'





class SubmissionColumns:

    id_seqpos = 'id_seqpos'

    reactivity = 'reactivity'

    deg_Mg_pH10 = 'deg_Mg_pH10'

    deg_pH10 = 'deg_pH10'

    deg_Mg_50C = 'deg_Mg_50C'

    deg_50C = 'deg_50C'



    @staticmethod

    def get_target_columns():

        return [SubmissionColumns.reactivity, SubmissionColumns.deg_Mg_pH10, SubmissionColumns.deg_pH10,

                SubmissionColumns.deg_Mg_50C, SubmissionColumns.deg_50C]





class Submission(SubmissionColumns):

    @staticmethod

    def MCRMSE(y_true, y_pred):

        colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

        return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)



    @staticmethod

    def get_submission_columns(optional=False):

        if optional:

            return SubmissionColumns.get_target_columns()

        else:

            return [SubmissionColumns.reactivity, SubmissionColumns.deg_pH10, SubmissionColumns.deg_Mg_50C]





class Definitions:

    @staticmethod

    def structure_tokens():

        return ['S', 'B', 'M', 'I', 'E', 'H', 'X']



    @staticmethod

    def nucleotides_tokens():

        return ['A', 'U', 'C', 'G']



    @staticmethod

    def tokens():

        n_tokens = Definitions.nucleotides_tokens()

        s_tokens = Definitions.structure_tokens()

        n_tokens.extend(s_tokens)

        return n_tokens



    @staticmethod

    def structures():

        return {

            'S': 'Paired Stem',

            'B': 'Bulge',

            'M': 'Multi Loop',

            'I': 'Internal Loop',

            'E': 'Dangling End',

            'H': 'Hairpin Loop',

            'X': 'External Loop'

        }



    @staticmethod

    def nucleotides():

        return {

            'A': 'adenine',

            'U': 'uracil',

            'C': 'cytosine',

            'G': 'guanine',

        }



# Data class 

class Data(Columns):

    def __init__(self, data_path, bpps_path=None):

        self.path = data_path

        self.bpps_path = bpps_path

        self.data = pd.read_json(self.path, lines=True)

        self.n_rows, self.n_cols = self.data.shape



    def get_sample(self):

        return self.data.iloc[[np.random.randint(low=0, high=self.n_cols)]]



    def get_bpps_matrix(self, id):

        bbps_file = id + '.npy'

        file_path = path.join(self.bpps_path, bbps_file)

        return np.load(file_path)



    def has_duplicates(self, column):

        return not self.data[column].is_unique



    def describe_data(self):

        print("Number of Data Samples: {}".format(self.n_rows))

        print("Number of Columns: {}".format(self.n_cols))

        # Added on need to basis

class SequenceAnalysis:

    @staticmethod

    def get_base_pairs_indices(structure):

        """

        takes the structure of the sequence and finds the indexes of matching brackets

        :param structure: str

        :return: list of tuples indicating start and end positions of matching brackets Ex: [(x,y), (x2,y2)]

        """

        paired_bases = list()

        stack = list()

        struct_len = len(structure)

        # Get the first occurrence of the left bracket

        i = 0

        while i < struct_len:

            if structure[i] == '(':

                stack.append(i)

                break

            i += 1



        # Move to the next char

        i += 1

        while i < struct_len:

            if structure[i] == '(':

                stack.append(i)

            elif structure[i] == ')':

                paired_bases.append((stack.pop(), i))

            else:

                # Ignore the '.' while looking for matching base pairs

                pass

            i += 1



        # stack is empty at the end if the above loop has no bugs

        if len(stack) > 0:

            raise AssertionError("Some bug while finding matching base pairs")



        # sort based on the first index on tuple

        paired_bases.sort(key=lambda x: x[0])

        return paired_bases



    @staticmethod

    def base_pair_counts(sequence, structure, similar=True):

        """

        accepts rna sequence re

        :param similar:

        :param structure:

        :param sequence:

        :return: list of tuples of format [(x, y, count)]

        """

        if not isinstance(sequence, str):

            raise TypeError('sequence is not of type "str"')



        if not isinstance(structure, str):

            raise TypeError('structure is not of type "str"')



        if not isinstance(similar, bool):

            raise TypeError('similar is not of type "bool"')



        bp_indices = SequenceAnalysis.get_base_pairs_indices(structure=structure)



        # convert indices to bases having entries (x, y) and (y,x) as different entries

        base_pairs_unmerged = [(sequence[i], sequence[j]) for i, j in bp_indices]



        bp_counts = Counter(base_pairs_unmerged)

        base_pairs_unmerged_counts = [(bp, bp_counts[bp]) for bp in bp_counts]



        if not similar:

            return base_pairs_unmerged_counts



        else:

            bases_count = len(Definitions.nucleotides())

            count_matrix = np.zeros(shape=(bases_count, bases_count), dtype = np.int64)

            base2int_dict = {base: ind for ind, base in enumerate(Definitions.nucleotides().keys())}

            int2base_dict = {num: base for base, num in base2int_dict.items()}





            # when similar is set to True , (x, y) and (y, x) are treated identically

            for i, item in enumerate(base_pairs_unmerged_counts):

                if item[0][0] > item[0][1]:

                    base_pairs_unmerged_counts[i] = ((item[0][1], item[0][0]), item[1])



            base_pairs_unmerged_counts.sort(key=lambda x: x[0][0])



            for base_pair, bp_count in base_pairs_unmerged_counts:

                b1, b2 = base_pair[0], base_pair[1]

                x = base2int_dict[b1]

                y = base2int_dict[b2]

                count_matrix[x][y] += bp_count

                count_matrix[y][x] = count_matrix[x][y]



            base_pairs_merged_counts = list()

            i = j = 0

            for i in range(bases_count):

                j = i+1

                while j < bases_count:

                    base_pairs_merged_counts.append(((int2base_dict[i], int2base_dict[j]), count_matrix[i][j]))

                    j += 1

        

            return base_pairs_merged_counts

    

    @staticmethod

    def overall_base_pairs_count(sequences, structures, similar=True, percent=True):

        if not isinstance(sequences, list):

            return TypeError('sequences is not of type "list"')



        if not isinstance(structures, list):

            return TypeError('structures is not of type "list"')



        total_bp_counts = list()



        for sequence, structure in zip(sequences, structures):

            total_bp_counts.extend(SequenceAnalysis.base_pair_counts(sequence, structure, similar=similar))



        bp_count_dict = dict()



        for bp, bp_count in total_bp_counts:

            if bp in bp_count_dict:

                bp_count_dict[bp] += bp_count

            else:

                bp_count_dict[bp] = bp_count



        if not percent:

            return bp_count_dict

        else:

            total_base_pairs = float(sum(bp_count_dict.values()))

            bp_percents_dict = {bp: round((bp_count / total_base_pairs) * 100, 2) for bp, bp_count in bp_count_dict.items()}

            return bp_percents_dict



    @staticmethod

    def base_distribution_by_position(sequences, seq_length, percent=True):

        """

        Accepts a list of sequences and returns the probabilities of the bases at each position

        :param sequences: list of strings

        :return: list keys: position indexes values: dict ('base_key': 'base_count')

        """

        total_sequences = len(sequences)

        sequence_matrix = list()

        distribution_dict = dict()

        token_dict = {base: 0 for base in Definitions.nucleotides_tokens()}



        for sequence in sequences:

            sequence_matrix.append(list(sequence[:seq_length]))



        # count the value counts at position i

        for pos in range(seq_length):

            distribution_dict[pos] = token_dict.copy()

            bases = [base[pos] for base in sequence_matrix]

            value_counts = Counter(bases)

            for base in value_counts.keys():

                if not percent:

                    distribution_dict[pos][base] = round(float(value_counts[base])/total_sequences, 2)

                else:

                    distribution_dict[pos][base] = round((float(value_counts[base])/total_sequences) * 100, 2)



        return distribution_dict



    @staticmethod

    def generate_base_pairing_probability(bpps_matrix):

        """

        Accepts a numpy 2D bpps matrix and generates the paring probabilites at each position

        It is as follows:

            for a position i, sum all the probabilities other than position i

        :param bpps_matrix:

        :return: 1d array represing base pairing probabilities at position i

        """

        if not isinstance(bpps_matrix, np.ndarray) or len(bpps_matrix.shape) != 2:

            raise TypeError("Expected numpy 2d array")



        # check if the bpps matrix is valid by checking the symetry

        is_valid = (bpps_matrix == bpps_matrix.transpose()).all()

        if not is_valid:

            raise ValueError("Expected Symmetric matrix")



        # base pairing itself probability is zero, assert the same

        # check if all diagonal elements in the matrix are zeros

        if (bpps_matrix.diagonal() != np.zeros(shape=bpps_matrix.shape[0])).all():

            raise ValueError("Expected diagonal elements to be all zeros")



        base_pair_probabilities = np.zeros(shape=bpps_matrix.shape[0])

        for index, row in enumerate(bpps_matrix):

            base_pair_probabilities[index] = np.sum(row)



        return base_pair_probabilities



class OneHotEncoder:

    token2int = dict()

    int2token = dict()



    def __init__(self, tokens):

        if not isinstance(tokens, list):

            raise TypeError('arg:tokens is not of type list')

        self.tokens = tokens



        for num, token in enumerate(self.tokens):

            OneHotEncoder.token2int[token] = num + 1

        # update int2token dictionary

        OneHotEncoder.int2token = {token: val for token, val in OneHotEncoder.token2int.items()}



    @staticmethod

    def conv_token2int(word):

            return [OneHotEncoder.token2int[char] for char in word]

        

            

    @staticmethod

    def conv_int2token(num_list, concat=True):

        if concat:

            return [OneHotEncoder.int2token[num] for num in num_list].join("")

        else:

            return [OneHotEncoder.int2token[num] for num in num_list]

class PositionalEncoder:

    @staticmethod

    def encode(sequence):

        """

        Implementation is loosely based on https://www.frontiersin.org/articles/10.3389/fgene.2019.00467/full

                    #

                    #  2, (if (Ri=A and Rj= U) or (Ri = U and Rj=A))

         P(Ri,Rj) = #  3,   (if (Ri= G and Rj= C) or (Ri = C and Rj=G))

                    #  x {0 < x < 2}  ,   (if (Ri= G and Rj= U) or (Ri= U and Rj=G))

                    #  0   else

                    #

        :param sequence: string

        :return: numpy 2d array of size (len(sequence) x len(sequence))

        """

        if not isinstance(sequence, str):

            raise TypeError("'sequence' is not of type 'str'")



        sequence_length = len(sequence)

        s = sequence

        weighted_matrix = np.zeros(shape=())

        i = j = 0

        for i in range(sequence_length):

            j = i+1

            while j < sequence_length:

                if (s[i] == 'A' and s[j] == 'U') or (s[i] == 'U' and s[j] == 'A'):

                    weighted_matrix[i][j] = weighted_matrix[j][i] = 2.

                elif (s[i] == 'G' and s[j] == 'C') or (s[i] == 'C' and s[j] == 'G'):

                    weighted_matrix[i][j] = weighted_matrix[j][i] = 3.

                elif (s[i] == 'G' and s[j] == 'U') or (s[i] == 'U' and s[j] == 'G'):

                    weighted_matrix[i][j] = weighted_matrix[j][i] = round(np.random.uniform(0, 2), 2)

                else:

                    # defaults to 0

                    pass

                # increment column index

                j += 1

        return weighted_matrix

class HelperFunctions:

    @staticmethod

    def print_feature_property(col_name, d_type, length=None):

        print('Feature: {} dtype: {} Length: {}'.format(col_name, d_type, length))



    @staticmethod

    def get_class_variables(class_name):

        return [attr for attr in dir(class_name) if not callable(getattr(class_name, attr)) and not attr.startswith("__")]

    

    

    @staticmethod

    def split_train_data(t_x, t_y, split_percent):

        # split the train and test data

        mask = np.random.rand(len(t_x)) < 0.8

        train_x = t_x[mask]

        train_y = t_y[mask]



        test_x = t_x[~mask]

        test_y = t_y[~mask]



        return train_x, train_y, test_x, test_y







    @staticmethod

    def keys_as_positions_to_labels(positional_values_dict):

        """

        Accepts a dictionary having positions as keys and values of multiple labels in value dict and converts them

        to a dict having labels in the keys and positional values in the array

        :return:

        """

        if not isinstance(positional_values_dict, dict):

            raise TypeError('arg: positional_values_dict is not of type dict')



        # convert the dict of dict to arrays

        values_dict = dict()

        for token in Definitions.nucleotides_tokens():

            values_dict[token] = list()



        for pos, value_dict in positional_values_dict.items():

            for base, base_val in value_dict.items():

                values_dict[base].append(base_val)



        return values_dict

    

    @staticmethod

    def plot_multi_line_positions_to_values_graph(positions, values_dict, ylabel='Percentage', title=None, fig_size=(25, 8)):

        # plot all the distributions on the same graph

        plt.figure(figsize=fig_size)

        if values_dict.keys()  == ['A', 'U', 'C', 'G']:

            order = ['A', 'U', 'C', 'G']

            for key in order:

                plt.plot(positions, values_dict[key], label=key)

        else:

            for key in values_dict.keys():

                plt.plot(positions, values_dict[key], label=key)

                

        plt.xlabel('Position of RNA Seq')

        plt.ylabel(ylabel=ylabel)

        plt.xticks(positions)

        plt.title(title)

        plt.legend(loc='upper right')



    @staticmethod

    def plot_stacked_bar_positions_to_values_graph(positions, values_dict, ylabel='Percentage', title=None, fig_size=(25, 10)):

        # plot all the distributions in stacked bar graph

        plt.figure(figsize=fig_size)

        pa = plt.bar(positions, values_dict['A'], label='A')

        pg = plt.bar(positions, values_dict['G'], label='G', bottom=np.array(values_dict['A']))

        pc = plt.bar(positions, values_dict['C'], label='C', bottom=(np.array(values_dict['A']) + np.array(values_dict['G'])))

        pu = plt.bar(positions, values_dict['U'], label='U',

                     bottom=(np.array(values_dict['A']) + np.array(values_dict['G']) + np.array(values_dict['C'])))

        plt.xlabel('Position of RNA Seq')

        plt.ylabel(ylabel=ylabel)

        plt.xticks(positions)

        plt.title(title)

        plt.legend((pa[0], pg[0], pc[0], pu[0]), ('A', 'G', 'C', 'U'), loc='upper right')



# create instance of training data and test data 



training_instance = Data(data_path=train_data_path, bpps_path= bpps_path)

test_instance = Data(data_path=test_data_path)



# Exploratory data analysis for training data



print("In training data:")

training_instance.describe_data()



# check for duplicate ids in training data

print('In training data: Duplicates found: {}'.format(training_instance.has_duplicates(column=Columns.id)))



# Exploratory data analysis for test data



print("In test data")

test_instance.describe_data()



# check for duplicate ids in training data

print('In test data: Duplicates found: {}'.format(training_instance.has_duplicates(column=Columns.id)))

train_data = training_instance.data



# Analyze the quality of the data for training instance

print("Training samples with (SN_ratio > 1): {}".format(train_data[train_data[Columns.signal_to_noise] > 1].shape[0]))

print("Training samples with (SN_ratio = 1): {}".format(train_data[train_data[Columns.signal_to_noise] == 1].shape[0]))

print("Training samples with (SN_ratio < 1): {}".format(train_data[train_data[Columns.signal_to_noise] < 1].shape[0]))



training_instance.noisy_data = train_data[train_data[Columns.signal_to_noise] < 1]

training_instance.data = train_data[train_data[Columns.signal_to_noise] > 1]

training_instance.data = training_instance.data.reset_index(drop=True)



# Analyze data with signal_to_noise > 1 and with SN_filter

print("Training samples with (SN_ratio > 1) and (SN_filter =1) : {}".format((training_instance.data[training_instance.data[Columns.SN_filter] ==1 ].shape[0])))

print("Training samples with (SN_ratio > 1) and (SN_filter =0) : {}".format((training_instance.data[training_instance.data[Columns.SN_filter] ==0 ].shape[0])))

                                            

# get the sample from training data

sample_train = training_instance.get_sample()



# get the coresponding bpps matrix

bpps_matrix = training_instance.get_bpps_matrix(id=sample_train[Columns.id].item())



# shape of bpps matrix

print('shape:{}'.format(bpps_matrix.shape))





# Check if the matrix is symmetric

print('Matrix is symmetric: {}'.format((bpps_matrix  == bpps_matrix.transpose()).all()))

feature_names = HelperFunctions.get_class_variables(Columns)

for feature in feature_names:

    feature_value = sample_train[feature].item()

    try:

        length = len(feature_value)

    except TypeError:

        length = None



    HelperFunctions.print_feature_property(feature, type(feature_value), length=length)



# get unique different sequence lengths in train data

print('Training data unique sequence lengths: {}'.format(training_instance.data[Columns.seq_length].value_counts()))



# get unique different sequence lengths in test data

print('Test data unique sequence lengths: {}'.format(test_instance.data[Columns.seq_length].value_counts()))

# Relation between bpps matrix and reactivity

train_sample = training_instance.get_sample()

reactivity_sample = train_sample[Columns.reactivity].item()

bpps_matrix_sample = training_instance.get_bpps_matrix(id=train_sample[Columns.id].item())

pairing_probabilities = SequenceAnalysis.generate_base_pairing_probability(bpps_matrix_sample)[:68]

positions = np.arange(0, len(pairing_probabilities))

HelperFunctions.plot_multi_line_positions_to_values_graph(positions=positions, values_dict= {'bpps': pairing_probabilities, 'reactivity': reactivity_sample},

                                                          ylabel='values', title='Compare Reactivity with Base pair probabilities for Random Sample')



# convert reactivities to matrices

reactivity_matrix = np.array(training_instance.data[Columns.reactivity].tolist(), dtype=np.float)

deg_50C_matrix = np.array(training_instance.data[Columns.deg_50C].tolist(), dtype=np.float)

deg_Mg_50C_matrix = np.array(training_instance.data[Columns.deg_Mg_50C].tolist(), dtype=np.float)

deg_pH10_matrix = np.array(training_instance.data[Columns.deg_pH10].tolist(), dtype=np.float)

deg_Mg_pH10_matrix = np.array(training_instance.data[Columns.deg_Mg_pH10].tolist(), dtype=np.float)



# convert reactivity errors to matrices

reactivity_error_matrix = np.array(training_instance.data[Columns.reactivity_error].tolist(), dtype=np.float)

deg_50C_error_matrix = np.array(training_instance.data[Columns.deg_error_50C].tolist(), dtype=np.float)

deg_Mg_50C_error_matrix = np.array(training_instance.data[Columns.deg_error_Mg_50C].tolist(), dtype=np.float)

deg_pH10_error_matrix = np.array(training_instance.data[Columns.deg_error_pH10].tolist(), dtype=np.float)

deg_Mg_pH10_error_matrix = np.array(training_instance.data[Columns.deg_error_Mg_pH10].tolist(), dtype=np.float)



# Add errors observed in reactivities to the experimental values

# Note that error = accepted_value - experimental value



reactivity_matrix = reactivity_matrix + np.negative(reactivity_error_matrix)

deg_50C_matrix = deg_50C_matrix + np.negative(deg_50C_error_matrix)

deg_Mg_50C_matrix = deg_Mg_50C_matrix + np.negative(deg_Mg_50C_error_matrix)

deg_pH10_matrix = deg_pH10_matrix + np.negative(deg_pH10_error_matrix)

deg_Mg_pH10_matrix = deg_Mg_pH10_matrix + np.negative(deg_Mg_pH10_error_matrix)

# compare mean reactivity and mean base pairing at every position 



mean_reactivity = reactivity_matrix.mean(axis=0)

pairing_probability_matrix = list()

for uid in training_instance.data[Columns.id]:

    bpps_matrix = training_instance.get_bpps_matrix(id=train_sample[Columns.id].item())

    pairing_probabilities = SequenceAnalysis.generate_base_pairing_probability(bpps_matrix_sample)[:68]

    pairing_probability_matrix.append(pairing_probabilities)



pairing_probability_matrix = np.array(pairing_probability_matrix)

mean_pairing_probability = pairing_probability_matrix.mean(axis=0)

positions = np.arange(0, len(mean_pairing_probability))



HelperFunctions.plot_multi_line_positions_to_values_graph(positions=positions, values_dict= {'mean_bpps': mean_pairing_probability, 'reactivity': mean_reactivity},

                                                          ylabel='values', title='Compare Mean Reactivity with Mean Base pair probabilities for Training Data')

# Find the correlation between mean reactivity and bpps 



df = pd.DataFrame(data={'mean_reactivity': mean_reactivity, 'mean_pair_prob': mean_pairing_probability})

positions = np.arange(0, len(mean_reactivity))

df['mean_reactivity'].corr(df['mean_pair_prob'])
pairing_probability_df = pd.DataFrame(data=pairing_probability_matrix)

reactivity_df = pd.DataFrame(data=reactivity_matrix)

corr = pairing_probability_df.corrwith(reactivity_df, axis=0).to_list()

positions = np.arange(0, len(corr))

HelperFunctions.plot_multi_line_positions_to_values_graph(positions=positions, values_dict={'corr': corr}, ylabel='corr', title='Position wise correlation between reactivity and pairing probabilities')
sample_structure = train_sample[Columns.structure].item()

base_pairs_indices = SequenceAnalysis.get_base_pairs_indices(structure=sample_structure)

print(sample_structure)

print(base_pairs_indices)

print(len(base_pairs_indices))

base_pairs_counts = SequenceAnalysis.base_pair_counts(sequence=train_sample[Columns.sequence].item(), structure=sample_structure, similar=False)

print('Base pair counts considering symmetry')

print(base_pairs_counts)



base_pairs_counts = SequenceAnalysis.base_pair_counts(train_sample[Columns.sequence].item(), sample_structure)

print('Base pair counts with out considering symmetry')

print(base_pairs_counts)



# Base pair counts for the training data

sequence_list = training_instance.data[Columns.sequence].tolist()

structure_list = training_instance.data[Columns.structure].tolist()

training_data_bp_percents = SequenceAnalysis.overall_base_pairs_count(sequence_list, structure_list)



# get 500 random sequences and corresponding structures

# generate 500 random indexes from train data

random_indexes = np.random.randint(low=0, high=training_instance.data.shape[0], size=10)

r_sequences = training_instance.data.loc[random_indexes, Columns.sequence].tolist()

r_structures = training_instance.data.loc[random_indexes, Columns.structure].tolist()

random_train_bp_percents = SequenceAnalysis.overall_base_pairs_count(r_sequences, r_structures)

### Relation between reactivity values before and after incubating with MG

a4_dims = (20, 12)

fig, axs = plt.subplots(2, 2, figsize=a4_dims)

pos_array = np.arange(68)



# Mean reactivities before and after incubating with Mg at 50C

axs[0][0].plot(pos_array, deg_50C_matrix.mean(axis=0), color='green', label='before')

axs[0][0].plot(pos_array, deg_Mg_50C_matrix.mean(axis=0), color='red', label='after')

axs[0][0].set_xlabel('Position of RNA Seq')

axs[0][0].set_ylabel('Reactivities')

axs[0][0].set_title('Mean Reactivity at 50C before and after incubating with Mg')

axs[0][0].legend(loc='upper right')



# Mean reactivities before and after incubating with Mg at pH10

axs[0][1].plot(pos_array, deg_pH10_matrix.mean(axis=0), color='green', label='before')

axs[0][1].plot(pos_array, deg_Mg_pH10_matrix.mean(axis=0), color='red', label='after')

axs[0][1].set_xlabel('Position of RNA Seq')

axs[0][1].set_ylabel('Reactivities')

axs[0][1].set_title('Mean Reactivity at pH10 before and after incubating with Mg')

axs[0][1].legend(loc='upper right')



# Relation between mean reactivity, reactivity_50C , reactivity at ph10 before incubating with Mg

axs[1][0].plot(pos_array, reactivity_matrix.mean(axis=0), color='green', label='experimental')

axs[1][0].plot(pos_array, deg_Mg_50C_matrix.mean(axis=0), color='red', label='50C')

axs[1][0].plot(pos_array, deg_Mg_pH10_matrix.mean(axis=0), color='blue', label='pH10')

axs[1][0].set_xlabel('Position of RNA Seq')

axs[1][0].set_ylabel('Reactivities')

axs[1][0].set_title('Mean Reactivity Comparison at Different Experimental conditions before incubating with Mg')

axs[1][0].legend(loc='upper right')



# Relation between mean reactivity, reactivity_50C , reactivity at ph10 after incubating with Mg

axs[1][1].plot(pos_array, reactivity_matrix.mean(axis=0), color='green', label='experimental')

axs[1][1].plot(pos_array, deg_50C_matrix.mean(axis=0), color='red', label='Mg_50C')

axs[1][1].plot(pos_array, deg_pH10_matrix.mean(axis=0), color='blue', label='Mg_pH10')

axs[1][1].set_xlabel('Position of RNA Seq')

axs[1][1].set_ylabel('Reactivities')

axs[1][1].set_title('Mean Reactivity Comparison at Different Experimental conditions after incubating with Mg')

axs[1][1].legend(loc='upper right')



plt.figure(figsize=(25, 8))

plt.plot(pos_array, np.exp(deg_Mg_50C_matrix.mean(axis=0) - deg_50C_matrix.mean(axis=0)), color='red', label='50C')

plt.plot(pos_array, np.exp(deg_Mg_pH10_matrix.mean(axis=0) - deg_pH10_matrix.mean(axis=0)), color = 'blue', label= 'pH10')

plt.xlabel('Position of RNA Seq')

plt.ylabel('Relative Degradation scaled with exp')

plt.title('Effect of Magnesium on LikelyHood Degradation')

plt.legend(loc='upper right')
# Get sequences  distribution by position

base_percentages = SequenceAnalysis.base_distribution_by_position(sequences=sequence_list, seq_length=67, percent=True)

values_dict = HelperFunctions.keys_as_positions_to_labels(base_percentages)

positions = list(base_percentages.keys())



HelperFunctions.plot_multi_line_positions_to_values_graph(positions, values_dict, title='Train Data: Overall percentage of Base at Position i')

HelperFunctions.plot_stacked_bar_positions_to_values_graph(positions, values_dict, title='Train Data: Overall Percentage of Each Base at Position i')



public_test_data = test_instance.data[test_instance.data[Columns.seq_length] == 107]

private_test_data = test_instance.data[test_instance.data[Columns.seq_length] == 130]



# plot for public test data 

public_test_data_bp_percents = SequenceAnalysis.base_distribution_by_position(sequences=public_test_data[Columns.sequence], seq_length=67, percent=True)

pub_test_positions = list(public_test_data_bp_percents.keys())

pub_test_values_dict = HelperFunctions.keys_as_positions_to_labels(positional_values_dict=public_test_data_bp_percents)

HelperFunctions.plot_multi_line_positions_to_values_graph(pub_test_positions, pub_test_values_dict, title='Public Test: Overall percentage of Base at Position i')

HelperFunctions.plot_stacked_bar_positions_to_values_graph(pub_test_positions, pub_test_values_dict, title='Public Test: Overall Percentage of Each Base at Position i')



# plot for private test data

private_test_data_bp_percents = SequenceAnalysis.base_distribution_by_position(sequences=private_test_data[Columns.sequence], seq_length=91, percent=True)

priv_test_positions = list(private_test_data_bp_percents.keys())

priv_test_values_dict = HelperFunctions.keys_as_positions_to_labels(positional_values_dict=private_test_data_bp_percents)

HelperFunctions.plot_multi_line_positions_to_values_graph(priv_test_positions, priv_test_values_dict, title='Private Test:Overall percentage of Base at Position i')

HelperFunctions.plot_stacked_bar_positions_to_values_graph(priv_test_positions, priv_test_values_dict, title='Private Test: Overall Percentage of Each Base at Position i')
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers as L



target_columns = Submission.get_submission_columns()

one_hot_encoder = OneHotEncoder(tokens=Definitions.tokens())



# encoding for train data

training_instance.data['seq_encoded'] = training_instance.data[Columns.sequence].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()

training_instance.data['loop_encoded'] = training_instance.data[Columns.predicted_loop_type].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()



# encoding for public test

public_test_data['seq_encoded'] = public_test_data[Columns.sequence].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()

public_test_data['loop_encoded'] = public_test_data[Columns.predicted_loop_type].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()



# encoding for private test

private_test_data['seq_encoded'] = private_test_data[Columns.sequence].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()

private_test_data['loop_encoded'] = private_test_data[Columns.predicted_loop_type].apply(lambda x: OneHotEncoder.conv_token2int(x)).tolist()



train_x = pd.DataFrame(list(zip(training_instance.data['seq_encoded'], training_instance.data['loop_encoded'])), columns= ['seq_encoded', 'loop_encoded'])

train_y = training_instance.data[Submission.get_submission_columns(optional=True) ]



train_x, train_y, validation_x, validation_y = HelperFunctions.split_train_data(train_x, train_y, split_percent=0.8)





def preprocess_inputs(df, cols=['seq_encoded', 'loop_encoded']):

    base_fea = np.transpose(

        np.array(

            df[cols]

            .applymap(lambda x: np.asarray(x).astype(np.float))

            .values

            .tolist()

        ),

        (0, 2, 1)

    )

    return base_fea





def preprocess_outputs(df, cols=Submission.get_submission_columns(optional=True)):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda x: np.asarray(x).astype(np.float))

            .values

            .tolist()

        ),

        (0, 2, 1)

    )



train_x = preprocess_inputs(train_x)

validation_x = preprocess_inputs(validation_x)



train_y = preprocess_outputs(train_y)

validation_y = preprocess_outputs(validation_y)



print('...............')

print(train_x.shape)

print(train_y.shape)

print(validation_x.shape)

print(validation_y.shape)

#train_x.loc[:, 'seq_encoded'] = train_x.seq_encoded.apply(lambda x: np.asarray(x[:67]).astype(np.float))



#train_x.loc[:, 'loop_encoded'] = train_x.loop_encoded.apply(lambda x: np.transpose(np.asarray(x[:67]).astype(np.float), (0, 2, 1)))



#validation_x.loc[:, 'seq_encoded'] = validation_x.seq_encoded.apply(lambda x: np.asarray(x[:67]).astype(np.float))



#validation_x.loc[:, 'loop_encoded'] = validation_x.loop_encoded.apply(lambda x: np.asarray(x[:67]).astype(np.float))





def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)







def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(

        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))





def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5,

                sp_dropout=0.2, embed_dim=15, hidden_dim=512, n_layers=7):

    inputs = L.Input(shape=(seq_len, 2))

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)



    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3])

    )

    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)



    for x in range(n_layers):

        hidden = gru_layer(hidden_dim, dropout)(hidden)



    # Since we are only making predictions on the first part of each sequence,

    # we have to truncate it

    truncated = hidden[:, :pred_len]

    out = L.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.optimizers.Adam(), loss=MCRMSE)



    return model



# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = build_model(embed_size=len(OneHotEncoder.token2int) + 1)



    model.summary()



checkpoint_path = 'vaccine.h5'

num_epochs = 100

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)



train_history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y),  epochs=num_epochs, verbose=2,

                    callbacks= [ tf.keras.callbacks.ReduceLROnPlateau(patience=5),

                                tf.keras.callbacks.ModelCheckpoint(checkpoint_path)

                    ])







x_positions = pd.DataFrame(data=np.arange(num_epochs), columns=['epochs'])

history_df = pd.DataFrame(data=train_history.history)

history_df = history_df.join(x_positions)

sns.lineplot(x="epochs", y='value', hue='variable', data=pd.melt(history_df, ["epochs"]))



model_public = build_model(seq_len=107, pred_len=107, embed_size=len(OneHotEncoder.token2int) + 1)

model_private = build_model(seq_len=130, pred_len=130, embed_size=len(OneHotEncoder.token2int) + 1)



model_public.load_weights(checkpoint_path)

model_private.load_weights(checkpoint_path)



public_test_data = preprocess_inputs(public_test_data)

private_test_data = preprocess_inputs(private_test_data)





public_preds = model_public.predict(public_test_data)

private_preds = model_private.predict(private_test_data)



public_test_ids = test_instance.data[test_instance.data[Columns.seq_length] == 107].id.tolist()

private_test_ids = test_instance.data[test_instance.data[Columns.seq_length] == 130].id.tolist()

print(len(public_test_ids))

print(len(private_test_ids))

print(len(public_test_ids) + len(private_test_ids) )

def generate_id_positions(ids, pos_len):

    positions = np.arange(0, pos_len)

    return np.array([ id + '_' + str(pos) for id in ids for pos in positions])



public_preds = public_preds.reshape(-1, 5)

private_preds = private_preds.reshape(-1, 5)



pub_test_id_seq_posdf =  pd.DataFrame(data = generate_id_positions(public_test_ids, 107), columns = [SubmissionColumns.id_seqpos])

priv_test_ids_seq_posdf = pd.DataFrame(data = generate_id_positions(private_test_ids, 130), columns = [SubmissionColumns.id_seqpos])



public_preds_df = pd.DataFrame(data= public_preds, columns=Submission.get_submission_columns(optional=True))

private_preds_df = pd.DataFrame(data= private_preds, columns=Submission.get_submission_columns(optional=True))



submission_pub_data = pd.concat([pub_test_id_seq_posdf, public_preds_df], axis=1)

submission_priv_data = pd.concat([priv_test_ids_seq_posdf, private_preds_df], axis=1)





submission_df = pd.concat([submission_pub_data, submission_priv_data], axis=0)

submission_df.to_csv('submission.csv', index=False)
