def get_stratified_folds_index(target, stratify_classes, n_splits=5, seed=50, shuffle=True):

    order = np.argsort(target)

    classbin = len(target) // stratify_classes

    stratify = np.argsort(np.argsort(target)) // classbin

    print('Prepare folds. Classbin:', classbin, 'Classes:', np.max(stratify)+1)



    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed).split(np.arange(len(target)), stratify))

    return folds