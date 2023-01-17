import random

from sklearn.experimental import enable_iterative_imputer
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.impute
import sklearn.metrics
import sklearn.model_selection
SCHEMA_PSYCHOTYPES = {
    'ENFJ': 1,
    'ENFP': 2,
    'ENTJ': 3,
    'ENTP': 4,
    'ESFJ': 5,
    'ESFP': 6,
    'ESTJ': 7,
    'ESTP': 8,
    'INFJ': 9,
    'INFP': 10,
    'INTJ': 11,
    'INTP': 12,
    'ISFJ': 13,
    'ISFP': 14,
    'ISTJ': 15,
    'ISTP': 16
}

def construct_features(source_df):
    """
    Construction of additional features.

    1. values of each of the 8 scales;
    2. psychotype (a set of four letters,
        each of which is chosen as the largest in a pair.
    """
    data = source_df.copy()

    # KPMI key matrix.
    questions_scales = [
        # q1
        [{'scale_e': 2}, {'scale_i': 2}],
        # q2
        [{'scale_s': 2}, {'scale_n': 2}],
        # q3
        [{'scale_f': 1}, {'scale_t': 1}],
        # q4
        [{'scale_j': 2}, {'scale_p': 2}],
        # q5
        [{'scale_e': 1}, {'scale_i': 2}],
        # q6
        [{'scale_n': 2}, {'scale_s': 1}],
        # q7
        [{'scale_f': 1}, {'scale_t': 2}],
        # q8
        [{'scale_j': 2}, {'scale_p': 1}],
        # q9
        [{'scale_e': 2}, {'scale_i': 2}],
        # q10
        [{'scale_s': 2}, {'scale_n': 2}],
        # q11
        [{'scale_f': 2}, {'scale_t': 2}],
        # q12
        [{'scale_p': 1}, {'scale_j': 1}],
        # q13
        [{'scale_i': 1}, {'scale_e': 2}],
        # q14
        [{'scale_s': 1}, {'scale_n': 2}],
        # q15
        [{'scale_t': 1}],
        # q16
        [{'scale_j': 2}, {'scale_p': 2}],
        # q17
        [{'scale_i': 1}, {'scale_e': 2}],
        # q18
        [{'scale_n': 1}, {'scale_s': 2}],
        # q19
        [{'scale_f': 2}, {'scale_t': 1}],
        # q20
        [{'scale_j': 1}, {'scale_p': 2}],
        # q21
        [{'scale_e': 2}, {'scale_i': 2}],
        # q22
        [{}, {'scale_s': 1}],
        # q23
        [{}, {'scale_t': 2}],
        # q24
        [{'scale_p': 1}, {'scale_j': 1}],
        # q25
        [{'scale_e': 1}, {'scale_i': 1}],
        # q26
        [{'scale_s': 1}, {'scale_n': 1}],
        # q27
        [{'scale_f': 1}, {'scale_t': 2}],
        # q28
        [{'scale_j': 1}, {'scale_p': 2}],
        # q29
        [{'scale_e': 1}, {'scale_i': 1}],
        # q30
        [{'scale_s': 2}, {'scale_n': 1}],
        # q31
        [{'scale_t': 2}, {'scale_f': 2}],
        # q32
        [{'scale_j': 1}, {'scale_p': 1}],
        # q33
        [{'scale_e': 1}, {'scale_i': 2}],
        # q34
        [{'scale_n': 2}, {'scale_s': 2}],
        # q35
        [{'scale_t': 2}, {'scale_f': 2}],
        # q36
        [{'scale_j': 2}, {'scale_p': 2}],
        # q37
        [{'scale_i': 1}, {'scale_e': 2}],
        # q38
        [{'scale_n': 2}, {'scale_s': 1}],
        # q39
        [{'scale_t': 1}, {'scale_f': 2}],
        # q40
        [{'scale_j': 2}, {'scale_p': 2}],
        # q41
        [{'scale_i': 1}, {'scale_e': 1}],
        # q42
        [{'scale_s': 1}, {'scale_n': 1}],
        # q43
        [{'scale_t': 1}, {'scale_f': 2}],
        # q44
        [{'scale_j': 2}, {'scale_p': 2}],
        # q45
        [{}, {'scale_i': 1}],
        # q46
        [{}, {'scale_s': 2}],
        # q47
        [{'scale_f': 1}, {'scale_t': 2}],
        # q48
        [{'scale_p': 1}, {'scale_j': 1}],
        # q49
        [{'scale_e': 1}, {'scale_i': 1}],
        # q50
        [{'scale_s': 2}],
        # q51
        [{'scale_t': 1}, {'scale_f': 1}],
        # q52
        [{'scale_j': 1}, {'scale_p': 1}],
        # q53
        [{'scale_e': 1}],
        # q54
        [{'scale_s': 2}],
        # q55
        [{'scale_t': 1}, {'scale_f': 1}],
        # q56
        [{}, {'scale_j': 1}],
        # q57
        [{'scale_e': 1}, {'scale_i': 1}],
        # q58
        [{'scale_s': 1}, {'scale_n': 1}],
        # q59
        [{'scale_t': 2}],
        # q60
        [{'scale_j': 2}, {'scale_p': 1}],
        # q61
        [{'scale_i': 1}, {'scale_e': 2}],
        # q62
        [{'scale_s': 1}, {'scale_n': 1}],
        # q63
        [{}, {'scale_t': 2}],
        # q64
        [{'scale_p': 1}],
        # q65
        [{'scale_e': 1}, {'scale_i': 2}],
        # q66
        [{'scale_s': 2}, {'scale_n': 1}],
        # q67
        [{}, {'scale_t': 2}],
        # q68
        [{'scale_p': 1}, {'scale_j': 1}],
        # q69
        [{'scale_e': 1}, {'scale_i': 2}],
        # q70
        [{}, {'scale_n': 2}],
        # q71
        [{}, {'scale_t': 2}],
        # q72
        [{'scale_j': 1}, {'scale_p': 2}],
        # q73
        [{'scale_e': 1}, {'scale_i': 2}],
        # q74
        [{'scale_n': 2}],
        # q75
        [{}, {'scale_t': 1}],
        # q76
        [{'scale_p': 1}, {'scale_j': 1}],
        # q77
        [{'scale_e': 1}, {'scale_i': 1}],
        # q78
        [{'scale_s': 1}],
        # q79
        [{'scale_f': 1}, {'scale_t': 1}],
        # q80
        [{'scale_p': 1}, {'scale_j': 1}],
        # q81
        [{'scale_e': 1}, {'scale_i': 2}],
        # q82
        [{'scale_s': 1}],
        # q83
        [{'scale_t': 2}],
        # q84
        [{'scale_j': 2}, {'scale_p': 1}],
        # q85
        [{'scale_e': 1}],
        # q86
        [{'scale_s': 1}, {'scale_n': 1}],
        # q87
        [{'scale_f': 1}],
        # q88
        [{}, {'scale_p': 1}],
        # q89
        [{'scale_e': 1}],
        # q90
        [{}, {'scale_s': 1}],
        # q91
        [{'scale_f': 1}, {'scale_t': 2}],
        # q92
        [{}, {'scale_p': 1}],
        # q93
        [{'scale_e': 1}],
        # q94
        [{'scale_n': 2}, {'scale_s': 1}],
        # q95
        [{'scale_f': 1}],
        # q96
        [{'scale_j': 1}],
        # q97
        [{'scale_e': 1}],
        # q98
        [{}, {'scale_s': 1}],
        # q99
        [{'scale_f': 1}],
        # q100
        [{'scale_j': 1}],
        # q101
        [{'scale_e': 1}],
        # q102
        [{'scale_s': 2}],
        # q103
        [{}, {'scale_f': 2}],
        # q104
        [{'scale_j': 1}],
        # q105
        [{'scale_e': 1}],
        # q106
        [{}, {'scale_n': 1}],
        # q107
        [{}, {'scale_f': 2}],
        # q108
        [{'scale_j': 1}],
        # q109
        [{'scale_e': 1}],
        # q110
        [{'scale_n': 1}],
        # q11
        [{}, {'scale_f': 1}],
        # q112
        [{'scale_j': 1}],
        # q113
        [{'scale_e': 1}],
        # q114
        [{}, {'scale_n': 1}],
        # q115
        [{}, {'scale_f': 1}],
        # q116
        [{'scale_j': 1}],
        # q117
        [{'scale_i': 1}],
        # q118
        [{'scale_n': 1}],
        # q119
        [{}, {'scale_f': 1}],
        # q120
        [{'scale_j': 1}],
        # q121
        [{}, {'scale_i': 1}],
        # q122
        [{}, {'scale_n': 1}],
        # q123
        [{'scale_f': 1}],
        # q124
        [{'scale_p': 1}],
        # q125
        [{'scale_i': 1}],
        # q126
        [{}, {'scale_n': 1}],
        # q127
        [{'scale_f': 1}],
        # q128
        [{'scale_p': 1}],
        # q129
        [{'scale_i': 1}],
        # q130
        [{}, {'scale_n': 1}],
        # q131
        [{}, {'scale_f': 1}],
        # q132
        [{'scale_p': 1}],
        # q133
        [{'scale_i': 1}],
        # q134
        [{}, {'scale_n': 1}],
        # q135
        [{'scale_f': 1}],
        # q136
        [{'scale_p': 1}],
        # q137
        [{'scale_i': 1}],
        # q138
        [{}, {'scale_n': 1}],
        # q139
        [{'scale_f': 1}],
        # q140
        [{'scale_p': 1}],
        # q141
        [{}, {'scale_n': 1}],
        # q142
        [{'scale_s': 1}],
    ]

    # Function for calculating the values of scales and psychotypes
    # for each questionnaire.
    def compute_scales(row):
        result = pd.Series(
            {
                'scale_e': 0,
                'scale_i': 0,
                'scale_s': 0,
                'scale_n': 0,
                'scale_t': 0,
                'scale_f': 0,
                'scale_j': 0,
                'scale_p': 0
            },
            dtype='uint16'
        )
        result['psychotype'] = ''
        # Calculation of scales.
        for i, _ in enumerate(questions_scales):
            # The actual index of the answer.
            ans_idx = int(row[f'q{i+1}'])
            # Getting the scales influenced by the answer chosen
            # in the question.
            ans_scales = questions_scales[i][ans_idx] \
                if ans_idx < len(questions_scales[i]) \
                else None
            # The answer does not affect the scales, let's move on.
            if not ans_scales:
                continue
            # We go through all the scales that are affected by the answer
            # and increase the value of each such scale.
            for key, value in ans_scales.items():
                result[key] += value

        # Psychotype.
        if result['scale_e'] > result['scale_i']:
            result['psychotype'] += 'E'
        else:
            result['psychotype'] += 'I'
        if result['scale_s'] > result['scale_n']:
            result['psychotype'] += 'S'
        else:
            result['psychotype'] += 'N'
        if result['scale_t'] > result['scale_f']:
            result['psychotype'] += 'T'
        else:
            result['psychotype'] += 'F'
        if result['scale_j'] > result['scale_p']:
            result['psychotype'] += 'J'
        else:
            result['psychotype'] += 'P'

        return result

    # Create a dataframe with calculated scales.
    scales = data.apply(compute_scales, axis=1, result_type='expand')
    # Присоединяем значения шкал к основному датасету.
    data = pd.concat([data, scales], axis=1)

    # Coding of respondents' psychotypes.
    data['psychotype'] = [SCHEMA_PSYCHOTYPES[p] for p in data['psychotype']]
    
    return data
seed = 19

# Loading data.
df = pd.read_csv(
    '/kaggle/input/kpmi-mbti-mod-test/kpmi_data.csv',
    low_memory=False,
    sep=';'
)

# Label encoding.
df['psychotype'] = [SCHEMA_PSYCHOTYPES[p] for p in df['psychotype']]

# We are not going to use data on profession and field of activity,
# so the columns with them can be deleted.
df.drop(['jobtitle', 'jobfield'], axis=1, inplace=True)

# Creation of training and test samples.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    df,
    df['satisfied'],
    stratify=df['satisfied'],
    test_size=.25,
)

# Removing a label from data.
x_train = x_train.drop(['satisfied'], axis=1).copy(deep=True)
x_test = x_test.drop(['satisfied'], axis=1).copy(deep=True)
model = sklearn.ensemble.GradientBoostingClassifier(random_state=seed)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f'Base model score: {score:.3f}')
trim_cnt = 82
for i in range(1, 50):
    # Trimming "extra" data before training the IterativeImputer.
    data = df.copy(deep=True)
    data.drop(
        [
            'scale_e',
            'scale_i',
            'scale_s',
            'scale_n',
            'scale_t',
            'scale_f',
            'scale_j',
            'scale_p',
            'psychotype',
            'satisfied'
        ],
        axis=1,
        inplace=True
    )

    imputer = sklearn.impute.IterativeImputer()
    imputer.fit(data)

    data = x_train.copy(deep=True)
    data.drop(
        [
            'scale_e',
            'scale_i',
            'scale_s',
            'scale_n',
            'scale_t',
            'scale_f',
            'scale_j',
            'scale_p',
            'psychotype'
        ],
        axis=1,
        inplace=True
    )

    # Zeroing data for questions to be cleared.
    trim_q_labels = [f'q{q}' for q in random.sample(range(1, 143), trim_cnt)]
    data[trim_q_labels] = np.NaN

    # Supplementing data for zeroed questions using an Imputer.
    data = imputer.transform(data)

    # Building a dataframe with supplemented questions.
    data = pd.DataFrame(
        np.rint(data),
        columns=[f'q{i}' for i in range(1, 143)],
        dtype=int
    )
    # Calculation of scales and psychotype.
    data = construct_features(data)

    # Building, training and evaluating the model on a cropped version.
    model = sklearn.ensemble.GradientBoostingClassifier(random_state=seed)
    model.fit(data, y_train)
    print(f'[{i}] score: {model.score(x_test, y_test):.3f}')