import numpy as np            
import pandas as pd           
import sklearn.ensemble       
import sklearn.metrics        
import sklearn.model_selection
def prepare_data(df):
    # Identify a label with less data.
    label_minor = df['satisfied'].value_counts().idxmin()
    # Forming of satisfied and unsatisfied datasets based on the amount of data
    # in each category.
    minor_df = df[df['satisfied'] == label_minor]
    major_df = df[df['satisfied'] != label_minor]
    # Calculating number of frames.
    parts_num = len(major_df.index)//len(minor_df.index)
    # Forming frames.
    frames = [
        major_df.iloc[i*len(minor_df.index):(i+1)*len(minor_df.index)]
        for i in range(parts_num+1)
    ]
    
    result = []

    for frame in frames:
        minor_part = minor_df
        # Since the last frame is most often smaller than the default size, 
        # we need to trim minor_part to actual size.
        if len(frame.index) < len(minor_df.index):
            minor_part = minor_part.iloc[:len(frame.index)]
        frame_data = pd.concat([frame, minor_part], ignore_index=False)

        frame_labels = frame_data['satisfied']
        frame_data.drop(
            ['jobfield', 'jobtitle', 'psychotype', 'satisfied'],
            axis=1,
            inplace=True
        )

        # Splitting data to train and validation part.
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
            frame_data,
            frame_labels,
            stratify=frame_labels,
            test_size=.2,
        )

        # Data standardization.
        mean = x_train.mean(axis=0)
        x_train -= mean
        std = x_train.std(axis=0)
        x_train /= std
        x_val -= mean
        x_val /= std

        result.append(
            {
                'x_train': x_train,
                'x_val': x_val,
                'y_train': y_train,
                'y_val': y_val
            }
        )

    return result
def fit_eval_model(source_df, filter_column, filter_value, iters_num=100):
    print(f'=== FILTER: {filter_column}/{filter_value} ===')
    df = source_df.copy(deep=True)
    # Trimming data to selected job type and job field.
    df = df.loc[df[filter_column] == filter_value]
    # Label encoding.
    df["psychotype_cat"] = df["psychotype"].astype('category').cat.codes
    lbl_vc = df["satisfied"].value_counts(sort=False)
    print(f'> Data size: {len(df.index)}, Satisfied {lbl_vc[1]}/{lbl_vc[0]}')

    avg_score_val = 0
    ensembles = []
    df, x_test, _, y_test = sklearn.model_selection.train_test_split(
        df,
        df['satisfied'],
        stratify=df['satisfied'],
        test_size=.15,
    )
    x_test.drop(
        ['jobfield', 'jobtitle', 'psychotype', 'satisfied'],
        axis=1,
        inplace=True
    )
    mean = x_test.mean(axis=0)
    x_test -= mean
    std = x_test.std(axis=0)
    x_test /= std

    print('Iters: ', end='', flush=True)
    for i in range(iters_num):
        print(f'{i+1}..', end='', flush=True)
        ensembles.append([])

        # Data preparation.
        data_frames = prepare_data(df)

        # Ensemble forming.
        for data in data_frames:
            model = sklearn.ensemble.GradientBoostingClassifier(
                learning_rate=.1,
                loss='deviance',
                max_depth=3,
                n_estimators=1000,
                random_state=None
            )
            model.fit(data['x_train'], data['y_train'])
        
            ensembles[-1].append({
                'score_val': model.score(data['x_val'], data['y_val']),
                'model': model
            })

        tmp = sum(model['score_val'] for model in ensembles[-1])
        avg_score_val += tmp/len(data_frames)
    print('done.')

    # Calculating and showing some statistical info.
    tmp = avg_score_val/iters_num
    print(f'\n> Average single model validation score:\t{tmp:.3f}')
    print(f'> Models in each ensemble:\t\t\t{len(ensembles[0])}')
    ensemble_score = 0
    for _, ensemble in enumerate(ensembles):
        for i, model in enumerate(ensemble):
            ensemble_score += model['model'].score(x_test, y_test)/len(ensemble)
    tmp = ensemble_score/len(ensembles)
    print(f'> Average ensemble test data score:\t\t{tmp:.3f}\n')
df = pd.read_csv(
    '/kaggle/input/kpmi-mbti-mod-test/kpmi_data.csv',
    low_memory=False,
    sep=';'
)

# Build models for...
fit_eval_model(df, 'jobfield', 'Staff and training')
fit_eval_model(df, 'jobfield', 'Communication, sales')
fit_eval_model(df, 'jobfield', 'Department head')