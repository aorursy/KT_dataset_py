def main(st , test_type):
    import pandas as pd
    import numpy as np
    df = pd.read_csv('../input/cervicalcancerrisk/kag_risk_factors_cervical_cancer.csv')
    df = df.replace('?', 0)
    
    test_list = ['Dx:Cancer', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy']
    test_list.remove(test_type)
    
    x = df.drop(test_list , axis = 1)
    y = df[test_type]
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = st)
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
    
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)
    
    from sklearn.decomposition import PCA
    pca=PCA(n_components=32)
    x_train=pca.fit_transform(x_train)
    x_test=pca.transform(x_test)
    explained_variance=pca.explained_variance_ratio_

    y_pred = classifier.predict(x_test)
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    
    return (ac*100)
main('mean','Hinselmann')
main('mean','Schiller')
main('mean','Citology')
main('mean','Biopsy')
