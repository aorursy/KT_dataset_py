%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for nicer factor plots
import matplotlib.pyplot as plt
def tasty_papaya_func(redness, firmness):
    r_dist = np.sqrt(np.square(redness)+np.square(firmness))
    return (r_dist<0.65) & (r_dist>0.45)
red_xx, firm_yy = np.meshgrid(np.linspace(0,1,40), np.linspace(0,1,40))
redness_x, firmness_y = [x.ravel() for x in [red_xx, firm_yy]]
tastiness_z = tasty_papaya_func(redness_x, firmness_y)
fig, ax1 = plt.subplots(1,1, figsize = (8,6))
ax1.contourf(red_xx[0,:], firm_yy[:,0], tastiness_z.reshape(red_xx.shape), cmap = 'viridis')
ax1.set_xlabel('Redness')
ax1.set_ylabel('Firmness')
ax1.set_title('Papaya Tastiness');
redness_samples, firmness_samples = np.random.uniform(0, 1, size = (2, 1000))
all_papaya_samples_df = pd.DataFrame(dict(redness = redness_samples, 
                  firmness = firmness_samples, 
                  tastiness = tasty_papaya_func(redness_samples, firmness_samples)))
train_df = all_papaya_samples_df.sample(100, random_state=2018)
plt.scatter(train_df['redness'], train_df['firmness'], c = train_df['tastiness'])
train_df.head(10)
from sklearn.svm import SVC
simple_svm = SVC(kernel = 'rbf', probability=True)
simple_svm.fit(train_df[['firmness', 'redness']], train_df['tastiness'])
fig, ax1 = plt.subplots(1,1, figsize = (8,6))
ax1.contourf(red_xx[0,:], firm_yy[:,0], 
             simple_svm.predict_proba(np.stack([redness_x, firmness_y],-1))[:,1].reshape(red_xx.shape), 
             cmap = 'viridis',
            vmin = 0,
            vmax = 0.75)

ax1.set_xlabel('Redness')
ax1.set_ylabel('Firmness')
ax1.set_title('Papaya Tastiness');
from sklearn.metrics import accuracy_score
def fit_and_show_model(model, 
                       cur_df, 
                       ax = None, 
                       title_str = 'Tastiness',
                       fit_model = True
                      ):
    if fit_model:
        model.fit(cur_df[['firmness', 'redness']], cur_df['tastiness'])
    pred_y_proba = model.predict_proba(np.stack([redness_x, firmness_y],-1))[:,1]
    model_accuracy = accuracy_score(tastiness_z, pred_y_proba>0.5)
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (8,6))
    ax.contourf(red_xx[0,:], firm_yy[:,0], 
             pred_y_proba.reshape(red_xx.shape), 
             cmap = 'viridis',
            vmin = 0,
            vmax = 0.75)
    ax.set_xlabel('Redness')
    ax.set_ylabel('Firmness')
    ax.set_title('%s, Accuracy %2.1f%%' % (title_str, 100*model_accuracy));
    return model, model_accuracy
fig, m_axs = plt.subplots(3, 3, figsize = (12, 16))
for c_ax, c_pts in zip(m_axs.flatten(), np.linspace(20, 400, 9).astype(int)):
    cur_svm = SVC(kernel = 'rbf', probability=True, random_state = 2018)
    fit_and_show_model(cur_svm, 
                       all_papaya_samples_df.sample(c_pts, random_state=2018), 
                       title_str = 'Sampled: {}'.format(c_pts),
                       ax = c_ax)
# initializing the learner
from modAL.models import ActiveLearner
initial_df = all_papaya_samples_df.sample(20, random_state=2018)
learner = ActiveLearner(
    estimator=SVC(kernel = 'rbf', probability=True, random_state = 2018),
    X_training=initial_df[['firmness', 'redness']], 
    y_training=initial_df['tastiness']
)
# query for labels
X_pool = all_papaya_samples_df[['firmness', 'redness']].values
y_pool = all_papaya_samples_df['tastiness'].values
query_idx, query_inst = learner.query(X_pool)
query_idx, query_inst
fig, m_axs = plt.subplots(2, 3, figsize = (12, 12))
last_pts = initial_df.shape[0]
queried_pts = []
for c_ax, c_pts in zip(m_axs.flatten(), np.linspace(20, 350, 6).astype(int)):
    for _ in range(c_pts-last_pts):
        query_idx, _ = learner.query(X_pool)
        queried_pts += [query_idx]
        learner.teach(X_pool[query_idx], y_pool[query_idx])
    last_pts = c_pts
    fit_and_show_model(learner, 
                       None, 
                       title_str = 'Sampled: {}'.format(c_pts),
                       ax = c_ax,
                       fit_model = False
                      )
guessed_pts_idx = np.array(queried_pts)
fig, ax1 = plt.subplots(1,1, figsize = (8,6))
ax1.imshow(tasty_papaya_func(redness_x, firmness_y).reshape(red_xx.shape)[::-1], 
            extent = [redness_x.min(), redness_x.max(), firmness_y.min(), firmness_y.max()],
             vmin = 0, vmax = 1,
             cmap = 'bone_r')
plt.colorbar(ax1.scatter(X_pool[guessed_pts_idx,0], 
            X_pool[guessed_pts_idx,1], 
            c = np.arange(guessed_pts_idx.shape[0]).reshape((-1, 1)),
           cmap = 'nipy_spectral'))
ax1.set_xlabel('Redness')
ax1.set_ylabel('Firmness')
ax1.set_title('Active Learning Sample Order ');
from sklearn.ensemble import RandomForestClassifier
fig, m_axs = plt.subplots(3, 3, figsize = (12, 16))
for c_ax, c_pts in zip(m_axs.flatten(), np.linspace(10, 400, 9).astype(int)):
    cur_rf = RandomForestClassifier(random_state = 2018)
    fit_and_show_model(cur_rf, 
                       all_papaya_samples_df.sample(c_pts, random_state=2018), 
                       title_str = 'Sampled: {}'.format(c_pts),
                       ax = c_ax)
learner = ActiveLearner(
    estimator=RandomForestClassifier(random_state = 2018),
    X_training=initial_df[['firmness', 'redness']], 
    y_training=initial_df['tastiness']
)
query_idx, query_inst = learner.query(X_pool)
query_idx, query_inst
fig, m_axs = plt.subplots(2, 3, figsize = (12, 12))
last_pts = initial_df.shape[0]
queried_pts = []
for c_ax, c_pts in zip(m_axs.flatten(), np.linspace(20, 350, 6).astype(int)):
    for _ in range(c_pts-last_pts):
        query_idx, _ = learner.query(X_pool)
        queried_pts += [query_idx]
        learner.teach(X_pool[query_idx], y_pool[query_idx])
    last_pts = c_pts
    fit_and_show_model(learner, 
                       None, 
                       title_str = 'Sampled: {}'.format(c_pts),
                       ax = c_ax,
                       fit_model = False
                      )
guessed_pts_idx = np.array(queried_pts)
fig, ax1 = plt.subplots(1,1, figsize = (8,6))
ax1.imshow(tasty_papaya_func(redness_x, firmness_y).reshape(red_xx.shape)[::-1], 
            extent = [redness_x.min(), redness_x.max(), firmness_y.min(), firmness_y.max()],
             vmin = 0, vmax = 1,
             cmap = 'bone_r')
plt.colorbar(ax1.scatter(X_pool[guessed_pts_idx,0], 
            X_pool[guessed_pts_idx,1], 
            c = np.arange(guessed_pts_idx.shape[0]).reshape((-1, 1)),
           cmap = 'nipy_spectral'))
ax1.set_xlabel('Redness')
ax1.set_ylabel('Firmness')
ax1.set_title('Active Learning Sample Order ');
def run_full_experiment(create_model_func,
                        data_df,
                        initial_pts,
                        sample_counts,
                        random_state,
                       n_instances=1):
    for i in range(20):
        initial_df = data_df.sample(initial_pts, random_state=random_state+1000*i)
        # ensure there is at least one positive case
        if initial_df['tastiness'].max():
            break
    learner = ActiveLearner(
        estimator=create_model_func(),
        X_training=initial_df[['firmness', 'redness']], 
        y_training=initial_df['tastiness']
    )
    X_pool = data_df[['firmness', 'redness']].values
    y_pool = data_df['tastiness'].values
    last_pts = initial_df.shape[0]
    results_list = []
    for c_pts in [x for x in sample_counts if x>=initial_pts]:
        for _ in range(0, c_pts-last_pts, n_instances):
            query_idx, _ = learner.query(X_pool, n_instances = n_instances)
            learner.teach(X_pool[query_idx], y_pool[query_idx])
        last_pts = c_pts
        
        try:
            pred_y_proba = learner.predict_proba(np.stack([redness_x, firmness_y],-1))[:,1]
        except IndexError as e:
            pred_y_proba = np.zeros_like(redness_x)
        model_accuracy = accuracy_score(tastiness_z, pred_y_proba>0.5)
        random_model = create_model_func()
        results_list+=[dict(model = 'Active Learning', accuracy = model_accuracy, points = c_pts, 
                            classifier = random_model.__class__.__name__)]
        for i in range(20):
            cur_df = data_df.sample(c_pts, random_state=random_state+1000*i)
            # ensure there is at least one positive case
            if cur_df['tastiness'].max():
                break
        random_model.fit(cur_df[['firmness', 'redness']], cur_df['tastiness'])
        
        try:
            pred_y_proba = random_model.predict_proba(np.stack([redness_x, firmness_y],-1))[:,1]
        except IndexError as e:
            pred_y_proba = np.zeros_like(redness_x)
        
        model_accuracy = accuracy_score(tastiness_z, pred_y_proba>0.5)
        results_list+=[dict(model = 'Random Sampling', accuracy = model_accuracy, points = c_pts,
                           classifier = random_model.__class__.__name__)]
    return pd.DataFrame(results_list)
from dask import bag # run the experiments in parallel
def run_multiple_experiments(n_exp, **kwargs):
    seq_iter = bag.from_sequence(range(n_exp)).map(lambda i: run_full_experiment(random_state = i, **kwargs))
    return pd.concat(seq_iter.compute())
sample_counts = np.linspace(20, 600, 9).astype(int)
%%time
rf_results = run_multiple_experiments(n_exp = 10, 
                                      create_model_func = lambda : RandomForestClassifier(random_state = 2018),
                    data_df = all_papaya_samples_df,
                    initial_pts = 20,
                   sample_counts = sample_counts)
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = rf_results, size = 6)
%%time
svm_results = run_multiple_experiments(n_exp = 8, 
                                       create_model_func=lambda :  SVC(kernel = 'rbf', probability=True, random_state = 2018),
                    data_df = all_papaya_samples_df,
                    initial_pts = 20,
                   sample_counts = sample_counts)
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = svm_results, size = 6)
%%time
svm_results_40 = run_multiple_experiments(n_exp = 8, 
                                       create_model_func=lambda :  SVC(kernel = 'rbf', probability=True, random_state = 2018),
                    data_df = all_papaya_samples_df,
                    initial_pts = 40,
                   sample_counts = sample_counts)
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = svm_results_40, size = 6)
%%time
svm_results_100 = run_multiple_experiments(n_exp = 8, 
                                       create_model_func=lambda :  SVC(kernel = 'rbf', probability=True, random_state = 2018),
                    data_df = all_papaya_samples_df,
                    initial_pts = 100,
                   sample_counts = sample_counts)
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = svm_results_100, size = 6)
%%time
from sklearn.neighbors import KNeighborsClassifier
kn_results = run_multiple_experiments(n_exp = 8, 
                                      create_model_func=lambda : KNeighborsClassifier(3),
                    data_df = all_papaya_samples_df,
                    initial_pts = 20,
                   sample_counts = sample_counts)
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = kn_results, size = 6)
all_results = pd.concat([rf_results, svm_results, svm_results_40, svm_results_100, kn_results])
sns.factorplot(x = 'points', y = 'accuracy',
               col = 'classifier',
               hue = 'model',
               data = all_results,
               size = 8
              )
nice_results = all_results.pivot_table(values = 'accuracy', 
                        index = ['points', 'classifier'], 
                        columns = 'model').reset_index()
nice_results['Active Boost (%)'] = 100*(nice_results['Active Learning']-nice_results['Random Sampling'])/nice_results['Random Sampling']
nice_results['Active Accuracy Boost (%)'] = 100*(nice_results['Active Learning']-nice_results['Random Sampling'])
nice_results.head(5)
sns.factorplot(x = 'points', y = 'Active Boost (%)', hue = 'classifier', data = nice_results, size = 6)
from keras.wrappers.scikit_learn import KerasClassifier # doesn't seem to work for this
from keras.models import Sequential
from keras.layers import Dense, Dropout
def build_simple_model():
    model = Sequential()
    model.add(Dense(4, input_shape = (2,)))
    model.add(Dense(8))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return model
%%time
from IPython.display import clear_output
dl_results = run_full_experiment(create_model_func=lambda: KerasClassifier(build_fn = build_simple_model, 
                                                                           epochs = 1, 
                                                                           verbose = 1),
                    data_df = all_papaya_samples_df,
                    initial_pts = 20,
                   sample_counts = sample_counts,
                                random_state = 0, 
                                n_instances = 25)
clear_output() # very noisy functions
sns.factorplot(x = 'points', y = 'accuracy', hue = 'model', data = dl_results, size = 6)
all_results = pd.concat([rf_results, svm_results, svm_results_40, svm_results_100, kn_results, dl_results])
sns.factorplot(x = 'points', y = 'accuracy',
               col = 'classifier',
               hue = 'model',
               data = all_results,
               size = 8
              )
nice_results = all_results.pivot_table(values = 'accuracy', 
                        index = ['points', 'classifier'], 
                        columns = 'model').reset_index()
nice_results['Active Boost (%)'] = 100*(nice_results['Active Learning']-nice_results['Random Sampling'])/nice_results['Random Sampling']
nice_results['Active Accuracy Boost (%)'] = 100*(nice_results['Active Learning']-nice_results['Random Sampling'])
nice_results.head(5)
