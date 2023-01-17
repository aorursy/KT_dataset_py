# Internet ON.



# !pip install -U pip

# !pip install evaluations





# Internet OFF. You need to add evaluations dataset (see input folders).



!pip install --no-deps '../input/evaluations/'
from evaluations.kaggle_2020 import row_wise_micro_averaged_f1_score
y_true = [

    'amecro',

    'amecro amerob',

    'nocall',

]

y_pred = [

    'amecro',

    'amecro amerob',

    'nocall',

]

row_wise_micro_averaged_f1_score(y_true, y_pred)
y_true = [

    'amecro',

    'amecro amerob',

    'nocall',

]

y_pred = [

    'amecro',

    'amecro bird666',

    'nocall',

]

row_wise_micro_averaged_f1_score(y_true, y_pred)
y_true = [

    'bird1 bird2 bird3 bird4',

    'bird1 bird2 bird3 bird4',

    'bird1 bird2 bird3 bird4',

]

y_pred = [

    'bird1 bird2 bird3 bird4',

    'bird6 bird7',

    'bird2 bird3 bird6 bird7',

]

row_wise_micro_averaged_f1_score(y_true, y_pred)