import gensim # word embedding interface library



import numpy as np # linear algebra



import os

import pickle # package to serialize/de-serialize python objects 



# Serialized version of the Goole News Word2vec Model from https://code.google.com/archive/p/word2vec/

WORD2VEC_MODEL_PATH = '/kaggle/input/google-news-word2vec-model/google_vecs.pickle'



# Load word embedding model from a serialized gensim KeyedVectors object

f_model = open(WORD2VEC_MODEL_PATH, 'rb')

model = pickle.load(f_model)

f_model.close()

# Load FacetE dataset



import json



FACETE_FILE_PATH_GEOGRAPHIC = '/kaggle/input/facete/FacetE_geographic.json'

FACETE_FILE_PATH_SPORTS = '/kaggle/input/facete/FacetE_sports.json'

FACETE_FILE_PATH_TECHNOLOGY = '/kaggle/input/facete/FacetE_technology.json'

FACETE_FILE_PATH_MISC = '/kaggle/input/facete/FacetE_misc.json'



f_facets = open(FACETE_FILE_PATH_GEOGRAPHIC, 'r')

facets_geographic = json.load(f_facets)

f_facets.close()



f_facets = open(FACETE_FILE_PATH_SPORTS, 'r')

facets_sports = json.load(f_facets)

f_facets.close()



f_facets = open(FACETE_FILE_PATH_TECHNOLOGY, 'r')

facets_technology = json.load(f_facets)

f_facets.close()



f_facets = open(FACETE_FILE_PATH_MISC, 'r')

facets_misc = json.load(f_facets)

f_facets.close()

# Tokenization 



# Trie to represent multi-words of the model's vocabulary

# (each node represent a token)

def build_vocab_tree(model):

    vocab_tree = dict()

    for elem in model.vocab:

        tokens = elem.split('_')

        current_tree = [vocab_tree, False]

        current_tokens = [t for t in tokens]

        for i, token in enumerate(tokens):

            if token in current_tree[0]:

                if i == (len(tokens) - 1):  # end of elem

                    current_tree[0][token][1] = True

                else:

                    current_tokens = current_tokens[1:]

                    current_tree = current_tree[0][token]

            else:

                current_tree[0][token] = [dict(), False]

                if i == (len(tokens) - 1):  # end of elem

                    current_tree[0][token][1] = True

                else:

                    current_tokens = current_tokens[1:]

                    current_tree = current_tree[0][token]

    return vocab_tree



# Tokenization function using the Trie to tokenize

def tokenize_string(vocab_tree, value):

        tokens = value.split('_')

        tokenization = []

        current_pos = 0

        while (current_pos < len(tokens)):

            longest_token = []

            current_token_pos = current_pos

            current_tree = [vocab_tree, False]

            current_token = []

            while (current_token_pos < len(tokens)) and (tokens[current_token_pos] in current_tree[0]):

                current_token.append(tokens[current_token_pos])

                next_tree = current_tree[0][tokens[current_token_pos]]

                if next_tree[1]:

                    longest_token = [token for token in current_token]

                    current_token_pos = current_pos + len(current_token)

                current_tree = next_tree

            if len(longest_token) > 0:

                tokenization.append('_'.join(longest_token))

            current_pos = max(current_token_pos, current_pos + 1)

        return tokenization



# Build Trie (vocab_tree) of vocabulary for tokenization



print('Building Trie ...')

vocab_tree = build_vocab_tree(model)

print('Done.')

import time



def format_term(vocab_tree, value):

    return tokenize_string(vocab_tree, value.replace(' ', '_'))



def get_vector(model, values):

    vectors = [model.get_vector(value) for value in values if value in model]

    if len(vectors) > 0:

        mean = np.mean(vectors, axis=0)

        return mean

    else:

        return None



def centeroidnp(model, entries):

    array = np.array([get_vector(model, entry) for entry in entries if (entry != []) and (entry[0] in model)])

    return np.mean(array, axis=0) if len(array) > 0 else None



def most_similar_to_given(source_vector, target_vectors, valid):

    id = np.argmax(target_vectors.dot(source_vector))

    if all(target_vectors[id] == valid):

        return True

    else:

        return False



def solve_analogy(model, centroid_h1, centroid_h2, assignment, target_vectors):

    most_similar_vector = centroid_h2 - centroid_h1 + get_vector(model, assignment[0])

    most_similar_vector /= np.linalg.norm(most_similar_vector)

    valid_vector = get_vector(model, assignment[1])

    valid_vector /= np.linalg.norm(valid_vector)

    return most_similar_to_given(most_similar_vector, target_vectors, valid_vector)



def run_analogy_evaluation(facets):



    analogy_results = []



    for header1 in facets:

        for header2 in facets[header1]:

            t_last = time.time()

            assignments = facets[header1][header2]

            headers = [header1, header2]

            assignments_formated = [[format_term(vocab_tree, v1), format_term(vocab_tree, v2)] for [

                v1, v2] in assignments]

            tlast = time.time()

            centroid_h1 = centeroidnp(model,

                [assignment[0] for assignment in assignments_formated])

            centroid_h2 = centeroidnp(model,

                [assignment[1] for assignment in assignments_formated])

            good_assignments = []

            bad_assignments = []

            missing_target = []

            missing_entity = []

            values = [assignment[1] for assignment in assignments_formated if (

                assignment[1] != []) and (assignment[1][0] in model)]

            target_vectors = np.array(

                [get_vector(model, target) for target in values])

            target_vectors = np.array(

                [target / np.linalg.norm(target) for target in target_vectors])

            tlast = time.time()

            tstart = tlast

            for i, assignment in enumerate(assignments_formated):

                # check if left part has a word embedding representation

                if (assignment[0] != []) and (assignment[0][0] in model):

                    # check if right part has a word embedding representation

                    if (assignment[1] != []) and (assignment[1][0] in model):

                        if solve_analogy(model, centroid_h1, centroid_h2, assignment, target_vectors):

                            good_assignments.append(

                                assignments[i])

                        else:

                            bad_assignments.append(

                                assignments[i])

                    else:

                        missing_target.append(

                            assignments[i])

                else:

                    missing_entity.append(assignments[i])

                if time.time() - tlast > 0.2:

                    tlast = time.time()

            print(header1 + ' -> ' + header2, ' Completed in %fs' % (time.time() - tstart))

            tlast = time.time()

            accuracy_present = len(

                good_assignments) / (len(good_assignments) + len(bad_assignments)) if (len(good_assignments) + len(bad_assignments)) > 0 else 0

            accuracy_present_entity = len(

                good_assignments) / (len(good_assignments) + len(bad_assignments) + len(missing_target)) if (len(good_assignments) + len(bad_assignments) + len(missing_target)) > 0 else 0

            accuracy_all = len(good_assignments) / (len(good_assignments) + len(

                bad_assignments) + len(missing_target) + len(missing_entity))

            coverage = (len(good_assignments) + len(bad_assignments)) / (len(

                good_assignments) + len(bad_assignments) + len(missing_target) + len(missing_entity))

            eval_report = {

                'id': i,

                'headers': headers,

                'good_assignments': good_assignments,

                'bad_assignments': bad_assignments,

                'missing_target': missing_target,

                'missing_entity': missing_entity,

                'accuracy_present': accuracy_present,

                'accuracy_present_entity': accuracy_present_entity,

                'accuracy_all': accuracy_all,

                'coverage': coverage

            }

            analogy_results.append(eval_report)



    return analogy_results



eval_results = dict()

eval_results['Geographic'] = run_analogy_evaluation(facets_geographic)

eval_results['Sports'] = run_analogy_evaluation(facets_sports)

eval_results['Technology'] = run_analogy_evaluation(facets_technology)

eval_results['Misc'] = run_analogy_evaluation(facets_misc)
print(eval_results['Technology'][0]['accuracy_all'])
import matplotlib.pyplot as plt

import matplotlib.patches as patches



def create_violine_plot(data, cov_data):



    # Create a figure instance

    fig, ax = plt.subplots()

    fig.subplots_adjust(right=0.8, left=0.15, bottom=0.10, top=0.95 )

    # combine these different collections into a list

    labels = list(data.keys())

    data_to_plot = [data[key] for key in labels]



    mean_values = [np.mean(data) for data in data_to_plot]



    # Create the boxplot

    for i, data in enumerate(data_to_plot):

        # calculate quartiles for box-plots

        quartile1, median, quartile3 = np.percentile(data, [25, 50, 75])

        # draw box-plots

        rect = patches.Rectangle((i+1-0.05,quartile1),0.1, quartile3-quartile1, linewidth=2,edgecolor='gray',facecolor=(0.5,0.5,0.5,0.2))

        ax.hlines(max(data), i+0.95, i+1.05, color='gray', linestyle='-', lw=2)

        ax.hlines(min(data), i+0.95, i+1.05, color='gray', linestyle='-', lw=2)

        ax.hlines(median, i+0.95, i+1.05, color='lightcoral', linestyle='-', lw=2)

        ax.vlines(i+1,quartile3, max(data), linewidth=1,edgecolor='gray',facecolor='none')

        ax.vlines(i+1, quartile1, min(data), linewidth=1,edgecolor='gray',facecolor='none')

        ax.plot(i+1, np.mean(data), 'rx')

        ax.add_patch(rect)

        

        # draw boxes for accuracy and coverage

        props_acc = dict(boxstyle='round', facecolor='white', alpha=0.5)

        props_cov = dict(boxstyle='round', facecolor='black', alpha=0.5)

        

        acc_text = ('%2.2f' % (float(round(100*mean_values[i],2)),) if mean_values[i] != 1 else '100.0') + '%'

        cov_text = ('%2.2f' % (float(round(100*np.mean(cov_data[labels[i]]),2)),) if np.mean(cov_data[labels[i]]) != 1 else '100.0') + '%'

        ax.text(0.05+ float(i)/len(data_to_plot), 0.85, acc_text, transform=ax.transAxes, fontsize=12,

                verticalalignment='top', bbox=props_acc, fontweight='normal')

        ax.text(0.05+ float(i)/len(data_to_plot), 0.75, cov_text, transform=ax.transAxes, fontsize=12,

                verticalalignment='top', bbox=props_cov, color='white')



    # draw violine plots

    bp = ax.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False)

    for part in bp['bodies']:

        part.set_facecolor('#CCDDCC')

        part.set_edgecolor('#00aa00')

        

    # print legend

    ax.text(1.05, 0.40, 'Accuracy', transform=ax.transAxes, fontsize=12,

            verticalalignment='top', bbox=props_acc)

    ax.text(1.05, 0.25, 'Coverage', transform=ax.transAxes, fontsize=12,

            verticalalignment='top', bbox=props_cov, color='white')

    ax.plot(4.7, 0.10, 'rx', clip_on=False)

    ax.text(1.10, 0.12, 'Mean', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.hlines(0.00, 4.65, 4.75, color='lightcoral', linestyle='-', lw=2, clip_on=False)

    ax.text(1.10, 0.02, 'Median', transform=ax.transAxes, fontsize=12, verticalalignment='top')



    # configure axes

    ax.set_xticks(np.arange(1, len(labels) + 1))

    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylim(0, 1)

    ax.set_xlim(0.5, 4.5)

    ax.tick_params(labelsize=16)



    ax.set_ylabel('Accuracy [%]', fontsize=14)



    # plot figure

    fig.set_size_inches(7,3.5)

    fig.set_dpi(100)

    plt.show()



def extract_data(eval_results, metric):

    data = []

    for report in eval_results:

        data.append(float(report[metric]))

    return data

    

# get data

data_points = dict()

cov_data = dict()

for category in eval_results:

    data_points[category] = extract_data(eval_results[category], 'accuracy_all')

    cov_data[category] = extract_data(eval_results[category], 'accuracy_all')



create_violine_plot(data_points, cov_data)


