def Snippet_200():
    print()
    print(format('How to find optimal parameters for CatBoost using GridSearchCV for Classification','*^82'))

    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform as sp_randFloat
    from scipy.stats import randint as sp_randInt
    from catboost import CatBoostClassifier

    # load the iris datasets
    dataset = datasets.load_wine()
    X = dataset.data; y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = CatBoostClassifier()
    parameters = {'depth'         : sp_randInt(4, 10),
                  'learning_rate' : sp_randFloat(),
                  'iterations'    : sp_randInt(10, 100)
                 }

    randm = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                               cv = 2, n_iter = 10, n_jobs=-1)
    randm.fit(X_train, y_train)

    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")
    print("\n The best estimator across ALL searched params:\n",randm.best_estimator_)
    print("\n The best score across ALL searched params:\n",randm.best_score_)
    print("\n The best parameters across ALL searched params:\n",randm.best_params_)
    print("\n ========================================================")

Snippet_200()
