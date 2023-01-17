import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from sklearn.model_selection import StratifiedKFold

from sklearn import model_selection
class Options :

  FastTestRead = False #reads 100 rows from file if set to true

  DrawEpochLoss = True #draws a graph of loss and accuracy of each epoch

  PlotAgeDiffDistribution = True

  RunCrossValidation = True

  

  FeatureFilter = []

  PrintAllChartPatterns = False



  TransformToIntegerScores = True

  AllowedOrb = 7

  MaximumScore = 10

  ScoreValidation = False



  TrainDataSize = 0.6

  TestDataSize = 0.2

  ValidationDataSize = (1 - TrainDataSize - TestDataSize)



  CrossValidationFoldCount = 5

  CrossValidationRandomState = 20180818

  CrossValidationShowAllFoldGraphs = True

  

  KerasDimensions = 405

  KerasEpochs = 100 if not FastTestRead else 10

  KerasBatchSize = 4500

  KerasModelCheckpointSave = True

  KerasEarlyStopping = False

  KerasReduceLearningRate = False

  KerasVerbose = 0

  KerasCheckPointForNEpoch = 30 #Save weights for every N epochs



  SkipColumns = 6 #skip first n columns from the file

  

  PrintChartPatternsForRow = -1 #-1 set row no 



  ReadNumberOfRowsFromFile = 100 if FastTestRead else None



  GlobalData = []
class StringUtil :

    @staticmethod

    def isBlank(sting):

      if isinstance(sting, str) and sting and sting.strip():

          #sting is not None AND sting is not empty or blank

          return False

      #sting is None OR sting is empty or blank

      return True
class AspectUtil :



    @staticmethod

    def get_binary_score(orb, feature_name) :

      if StringUtil.isBlank(orb) :

        return 0



      return float(orb)





    @staticmethod

    def get_aspect_score(orb, feature_name):

      if StringUtil.isBlank(orb) :

        return 0



      forb = float(orb)



      if (abs(forb) > Options.AllowedOrb) :

        return 0



      MaxMeaningfulOrb = 7



      result = -(float(Options.MaximumScore) / float(MaxMeaningfulOrb)) * abs(float(orb)) + Options.MaximumScore

      result = abs(result)

      return result
class DataValidation:

    

    @staticmethod

    def validation(dsConcat, X, y):

        DataValidation.printChartPatternsForValidation(dsConcat, Options.PrintChartPatternsForRow)



    @staticmethod

    def printChartPatternsForValidation(dataset, rowNo):

        if (rowNo <= 0):

          return



        print(dataset.iloc[rowNo,0])

    

        for i in range(len(dataset.columns)) :

            if (isBlank(str(dataset.iloc[rowNo, i])) == False) :

                print(dataset.columns[i] + " " + str(dataset.iloc[rowNo, i]))



    @staticmethod

    def getPatternIndex(dataset, patternName):

        for i in range(len(dataset)):

            columnName = dataset[i]

            if (columnName.find(patternName) != -1) :

              return i

        return -1





    @staticmethod

    def getPatternIndexes(dataset, patternName):

        indexes = []



        for i in range(len(dataset)):

            columnName = dataset[i]

            if (columnName.find(patternName) != -1) :

              indexes.append(i)



        return indexes





    @staticmethod

    def unimportantFeatures(Xa, y, feature_names):



        resultIndexes = []



        for i in range(len(feature_names)):

            pattern = feature_names[i]

            posPatternCount, negPatternCount = DataValidation.printChartPatternCounts(Xa, y, feature_names, pattern)

            diffRatio = 0



            if (negPatternCount != 0):

              diffRatio = ((posPatternCount - negPatternCount) * 100) / (negPatternCount)

              diffRatio = abs(diffRatio)



            if (diffRatio < 1) :

              resultIndexes.append(i)



        return resultIndexes



    @staticmethod

    def processAllChartPatternCountDifferences(Xa, y, feature_names, toPrint):



        resultDict = {}

        resultDetailed = {}



        for i in range(len(feature_names)):

            pattern = feature_names[i]

            posPatternCount, negPatternCount = DataValidation.printChartPatternCounts(Xa, y, feature_names, pattern)

            diffRatio = 0



            if (negPatternCount != 0):

              diffRatio = ((posPatternCount - negPatternCount) * 100) / (negPatternCount)

              diffRatio = abs(diffRatio)



            resultDetailed[pattern] = { 'diffRatio' : diffRatio, 'posPatternCount' : posPatternCount, 'negPatternCount' : negPatternCount }

            resultDict[pattern] = diffRatio



        sorted_dicdata = sorted(resultDict.items(), key = operator.itemgetter(1), reverse = True)



        if (toPrint) : 

          for item in sorted_dicdata:

            print(item[0], item[1])



          for key, value in resultDetailed.items():

            print(key, "diffRatio", value.get('diffRatio'), "posPatternCount: ", value.get('posPatternCount'), "negPatternCount: ", value.get('negPatternCount'))



        return sorted_dicdata



    @staticmethod

    def printChartPatternCounts(Xa, y, feature_names, patternName):

        posCount = 0 #positive group pattern count

        negCount = 0 #negative group pattern count

        i = DataValidation.getPatternIndex(feature_names, patternName)

        rowCount = len(Xa)



        if (isinstance(Xa, (list, tuple, np.ndarray)) == False) :

          for j in range(rowCount):

              if (Xa.iloc[j, i] == 1):

                  colValue = y[j]

            

                  if (colValue == 1):

                    posCount += 1

                  elif (colValue == 0):

                    negCount += 1

        else :

          for j in range(rowCount):

              if (isBlank(Xa[j][i]) == False):

                  colValue = y[j]

            

                  if (colValue == 1):

                    posCount += 1

                  elif (colValue == 0):

                    negCount += 1



        return posCount, negCount





    @staticmethod

    def plot_age_diff_distribution(dsConcat) :

      if (Options.PlotAgeDiffDistribution == False) :

        return



      age_diffs = dsConcat["BMinusAAgeDifferenceYears"].values

      age_len = len(age_diffs)

      half = int(age_len / 2)

      

      # Plot the histograms

      fig, ax = plt.subplots()

      ax.hist(age_diffs[0:half], bins = 100, alpha=0.5, label='Real couples')

      ax.hist(age_diffs[half:age_len], bins = 100, alpha=0.3, label='Fake couples')

      ax.set_xlabel('Age difference (MaleAge-FemaleAge)')

      ax.set_ylabel('Chart count')

      plt.legend(loc='best')

      plt.show()

      return







    @staticmethod

    def drawOrb(Xa, y):

        return

        toDrawPositive = []

        toDrawNegative = []

        patternName = 'AB-VenCnjMar'



        i = DataValidation.getPatternIndex(feature_names, patternName)

        rowCount = len(Xa)



        for j in range(rowCount):

            if (isBlank(Xa[j][i]) == False):

                colValue = y[j]

            

                if (colValue == 1):

                  toDrawPositive.append(float(Xa[j][i]))

                elif (colValue == 0):

                  toDrawNegative.append(float(Xa[j][i]))



        plt.hist(toDrawNegative, bins= 10)  

        plt.title("Histogram")

        plt.axis([-10, 10, 0, 40])

        plt.show()







    @staticmethod

    def print_score(is_real, pattern_name, data_txt, curr_score, aspect_score, coefficient):

      if (Options.ScoreValidation == False or curr_score == 0) :

        return



      print ( ("Real:%s  Pattern:%s  CurrScore:%f  Txt:%s  AspectScore:%f  Coeff:%f") % (is_real, pattern_name, curr_score, data_txt, aspect_score, coefficient) )





    @staticmethod

    def print_data_summary(msg, X_train, y_train, X_validate, y_validate, X_test, y_test) :

      print(msg)

      print(("Train rows: %i      Validate rows: %i      Test rows: %i") % (len(X_train), len(X_validate), len(X_test)) )

      print(("Train columns: %i   Validate columns: %i   Test columns: %i") % (len(X_train[0]), len(X_validate[0]), len(X_test[0])) )

import time

import pandas as pd



class DataReader :

    

    @staticmethod

    def readDataSet(filename, positives):

        pds = pd.read_csv(filename, '\t', header=0, nrows=Options.ReadNumberOfRowsFromFile)

        rowCount = len(pds['Chart_A_Name'])

        newColVal = 1 if positives else 0

        pds['Result'] = pd.Series(newColVal, index=pds.index)

        return pds, rowCount

    

    @staticmethod

    def combineDataSet(filesArray):

        dsPositives = DataReader.readDataSet(filesArray['positives'], True)

        dsRndNegatives = DataReader.readDataSet(filesArray['negatives'], False)

        frames = [dsPositives[0], dsRndNegatives[0]]

        dsConcat = pd.concat(frames)

        return dsConcat, dsPositives, dsRndNegatives





    @staticmethod

    def columnFiltered(colName, filterList):

      if StringUtil.isBlank(colName):

          return True

    

      for i in range(len(filterList)):

         if colName.find(filterList[i]) != -1 :

          return True

    

      return False





    @staticmethod

    def fixColumnNames(ds):

      for i in range(len(ds.columns.values)):

          ds.columns.values[i] = ds.columns.values[i].replace("_", "")

          ds.columns.values[i] = ds.columns.values[i].replace(" ", "-")





    @staticmethod

    def filterColumns(ds, featureFilterList):

      colToFilter = []

      for i in range(len(ds.columns.values)):

          if DataReader.columnFiltered(ds.columns.values[i], featureFilterList):

              colToFilter.append(i)

      return colToFilter





    @staticmethod

    def satisfyOrb(orb) : 

      if (Options.AllowedOrb >= abs(float(orb))) :

        return True

      else:

        return False







    #poor performance code to retain column names (using arrays would be faster)

    @staticmethod

    def xDataSetToBinaryScores(X):

        for i in range(len(X.values)):

            for j in range(len(X.iloc[i])):

                if (StringUtil.isBlank(X.iat[i, j]) or isAspect(X.iat[i, j], X.values[i])):

                    X.iat[i,j] = 0

                else:

                    X.iat[i,j] = 1



    @staticmethod

    def xArrayToBinaryScores(X, feature_names):

        for j in range(len(feature_names)):

            for i in range(len(X)):

                if (StringUtil.isBlank(X[i, j])):

                    X[i, j] = 0

                elif (DataReader.satisfyOrb(X[i, j]) == False):

                    X[i, j] = 0

                else:

                    X[i, j] = 1



    @staticmethod

    def xArrayToAspectScores(X, feature_names):

        for j in range(len(feature_names)):

            for i in range(len(X)):

                X[i, j] =  AspectUtil.get_aspect_score(X[i, j], feature_names[j])

                    





    @staticmethod

    def transform_to_scores(X_train, X_validate, X_test, feature_names, cast_as_type = 'int64'):

      if (Options.TransformToIntegerScores) :

        print("Transforming to ", cast_as_type, " scores.")

        DataReader.xArrayToAspectScores(X_train, feature_names)

        DataReader.xArrayToAspectScores(X_validate, feature_names)

        DataReader.xArrayToAspectScores(X_test, feature_names)

      else :

        print("Transforming to binary scores.")

        DataReader.xArrayToBinaryScores(X_train, feature_names)

        DataReader.xArrayToBinaryScores(X_validate, feature_names)

        DataReader.xArrayToBinaryScores(X_test, feature_names)



      X_train = X_train.astype(cast_as_type)

      X_validate = X_validate.astype(cast_as_type)

      X_test = X_test.astype(cast_as_type)



      return X_train, X_validate, X_test





    @staticmethod

    def convert_to_array(dataframe, convert_features) :

      feature_names = None

      columnLen = dataframe.shape[1]



      if (convert_features) : 

        feature_names = dataframe.columns[(Options.SkipColumns + 1):columnLen-1]

      

      Xa = dataframe.iloc[:, (Options.SkipColumns + 1):columnLen-1].values

      y = dataframe.iloc[:, columnLen - 1].values



      return Xa, y, feature_names





    #contains all aux methods after dataset read and before score conversion

    @staticmethod

    def prepare_dataset(fileNames):

        start_time = time.time()

    

        dsConcat, dsPositives, dsRndNegatives = DataReader.combineDataSet(fileNames)

        colsToFilter = DataReader.filterColumns(dsConcat, Options.FeatureFilter)

        dsConcat.drop(dsConcat.columns[colsToFilter], axis=1, inplace=True)

        DataReader.fixColumnNames(dsConcat)

        keepAsDataSet = False



        columnLen = dsConcat.shape[1]



        feature_names = None

    

        if (keepAsDataSet):

            X = dsConcat.iloc[:, (Options.SkipColumns + 1):columnLen]

            y = dsConcat.iloc[:, columnLen - 1]

            ceateFeatureMap(X.columns)

            DataReader.xDataSetToBinaryScores(X)

        else: #convert to array for performance

            Xa, y, feature_names = DataReader.convert_to_array(dsConcat, True)

            

            if (Options.PrintAllChartPatterns) : #placed here because of array processing

              sorted_dicdata = DataValidation.processAllChartPatternCountDifferences(Xa, y, feature_names, True)

            

            DataValidation.drawOrb(Xa, y)



            if (Options.TransformToIntegerScores) :

              DataReader.xArrayToAspectScores(Xa, feature_names)

            else :

              DataReader.xArrayToBinaryScores(Xa, feature_names)

        

            X = pd.DataFrame(data=Xa, columns=feature_names)

    

    

        elapsed_time = time.time() - start_time

        print(("%f seconds elapsed during dataset read.") % elapsed_time)

    

        return dsConcat, dsPositives, dsRndNegatives, X, y, feature_names
import numpy as np

import matplotlib.pyplot as plt

import itertools





from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix







class ClassifierInput:



  def __init__(self, no, name, lib_type, creator) :

      self.no = no

      self.name = name

      self.library_type = lib_type

      self.creator = creator





  def allocate(self):

      self.classifier = self.creator()

      return self





  def set_classification_params(self, paramz) :

      self.classificationParams = paramz



      if (self.library_type == "sklearn") :

        self.classifier.set_params(**paramz)





  def set_vectors(self, dfConcat, dfPositives, dfRndNegatives, X, y, feature_names) :

      self.X = X

      self.y = y

      self.dfConcat = dfConcat #Concatenated dataframe of dsPositives and dsNegatives

      self.dfPositives = dfPositives #Only Positive samples

      self.dfRndNegatives = dfRndNegatives #Only Negative samples

      self.feature_names = feature_names #Feature names





  def shuffle(self) : 

    self.dfConcat = self.dfConcat.sample(frac=1).reset_index(drop=True)

    self.X, self.y, self.feature_names = DataReader.convert_to_array(self.dfConcat, True)





  def predict(self, X_test, y_test, dataset_name='') :

    y_pred = self.classifier.predict(X_test)

    predictions = [np.round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy On Test Data %s : %.2f%%" % (dataset_name, accuracy * 100.0))

    return accuracy, y_pred





  def predict_proba(self, X_test) :

    y_pred_prob = self.classifier.predict_proba(X_test)

    return y_pred_prob







  # An area under the ROC curve of value A, for example means that, a randomly

  # selected case from the group

  # with the target equals Y has a score larger than that for a randomly chosen

  # case from the group with

  # the target equals N in A% of the time.

  def draw_roc_curve(self, X_test, y_test, title_prefix='') :

    y_pred_prob = self.predict_proba(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_prob[:, 1])

    roc_auc = auc(false_positive_rate, true_positive_rate)



    plt.figure(1)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Luck', alpha=.8)

    plt.plot(false_positive_rate, true_positive_rate, label=title_prefix + self.name + ' (area = {:.3f})'.format(roc_auc))

    plt.xlabel('False positive rate (1-specificity)')

    plt.ylabel('True positive rate (sensitivity)')

    plt.title('ROC curve')

    plt.legend(loc='best')

    plt.show()







  def confusion_matrix(self, y_test, y_pred) :

    cnf_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    return cnf_matrix





  def draw_confusion_matrix_data(self, y_test, y_pred, class_names, title, normalize=False) :

    cnf_matrix = self.confusion_matrix(y_test, y_pred)

    self.draw_confusion_matrix(cnf_matrix, classes=class_names, normalize = normalize,

                      title='Confusion matrix')



  def draw_confusion_matrix(self, cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix')



    print(cm)



    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()







  def draw_cv_mean(self, cv_accuracy) :

    x = np.arange(1,len(cv_accuracy) + 1,1)

    y = cv_accuracy    

    y_mean = [np.mean(y)] * len(x)

    

    fig,ax = plt.subplots()

    # Plot the data

    data_line = ax.plot(x,y, label='CV Fold Accuracy', marker='o')

    # Plot the average line

    mean_line = ax.plot(x,y_mean, label='Mean', linestyle='--')

    legend = ax.legend(loc='upper right')

    plt.show()





  def print_cv_details(self, cv_accuracy) :

    print("\nCross validation average accuracy -> %.2f%% (STD : +/- %.2f%%)" % (100 * np.mean(cv_accuracy), 100 * np.std(cv_accuracy)))



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import keras

import os.path



from keras.callbacks import ModelCheckpoint





def baseline_model() :

    from keras.models import Sequential

    from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization

    from keras.constraints import max_norm

    from keras.optimizers import RMSprop

    model = Sequential()

    model.add(Dropout(0.42, input_shape=(Options.KerasDimensions,)))

    model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))

    model.add(Dropout(0.79))

    rms = RMSprop(lr = 0.00050)

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

    return model







class NeuralNetClassifier(ClassifierInput): #class to prevent exception when there is no keras installation

  def allocate(self) :

    import tensorflow as tf

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)

    from keras.wrappers.scikit_learn import KerasClassifier

    self.creator = KerasClassifier

    self.classifier = self.creator(build_fn=baseline_model, epochs=Options.KerasEpochs, batch_size=Options.KerasBatchSize, verbose=Options.KerasVerbose)

    return self







  def fit(self, X_train, y_train, **kwargs) :



    Options.KerasDimensions = len(X_train[0])

        

    callbacks = []

    

    saved_files = os.listdir('.')

    for item in saved_files:

      if item.endswith(".hdf5"):

        os.remove(os.path.join('.', item))

    

    filepath="nn_weights-{epoch:02d}.hdf5"



    if (Options.KerasModelCheckpointSave) : 

      checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=Options.KerasVerbose, save_weights_only=False, save_best_only=False, mode='max', period = Options.KerasCheckPointForNEpoch)

      callbacks.append(checkpoint)



    if (Options.KerasEarlyStopping) : 

      early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=Options.KerasVerbose, mode='auto')

      callbacks.append(early_stop)



    if (Options.KerasReduceLearningRate) : 

      reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose = Options.KerasVerbose)

      callbacks.append(reduce_lr)



    self._fitted = self.classifier.fit(X_train, y_train, batch_size = Options.KerasBatchSize, epochs = Options.KerasEpochs, verbose = Options.KerasVerbose,

          validation_data = kwargs['validation_data'], callbacks=callbacks)



    print("Fit done.")

    return self._fitted



   

  def print_summary(self) :

    return





  def draw_epoch_loss(self, X_test, y_test):

    if Options.DrawEpochLoss == False :

      return



    #from neuralnetclassifier import baseline_model

    temp_test_model = baseline_model()

    test_over_time = []

    test_result = []



    for i in range(len(self._fitted.history['acc'])):

      saved_file = "nn_weights-%02d.hdf5" % ((int(i)+Options.KerasCheckPointForNEpoch))

      if i % Options.KerasCheckPointForNEpoch == 0 and os.path.isfile(saved_file) : 

          temp_test_model.load_weights(saved_file)

          test_result = temp_test_model.evaluate(X_test, y_test, verbose=Options.KerasVerbose)

          # 0 is loss; 1 is accuracy

          test_over_time.append(test_result)

      else :

          test_over_time.append(test_result)



    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(range(len(self._fitted.history['loss'])), self._fitted.history['loss'],linestyle='-', color='blue',label='Training', lw=2)

    ax1.plot(range(len(np.array(test_over_time)[:,0])), np.array(test_over_time)[:,0], linestyle='-', color='green',label='Test', lw=2)

    ax2.plot(range(len(self._fitted.history['acc'])), self._fitted.history['acc'],linestyle='-', color='blue',label='Training', lw=2)

    ax2.plot(range(len(np.array(test_over_time)[:,1])), np.array(test_over_time)[:,1], linestyle='-', color='green',label='Test', lw=2)

    leg = ax1.legend(bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0.,fontsize=13)

    ax1.set_xticklabels('')

    #ax1.set_yscale('log')

    ax2.set_xlabel('# Epochs',fontsize=14)

    ax1.set_ylabel('Loss',fontsize=14)

    ax2.set_ylabel('Accuracy',fontsize=14)

    plt.show()



  


from sklearn.feature_selection import RFE

#from sklearn.linear_model import RandomizedLasso

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC





class Classification :



    ClassifiersToRun = [ 11 ]



    Classifiers = [

                   ClassifierInput(1, "LR", "sklearn", LogisticRegression),

                   ClassifierInput(2, 'RF', "sklearn", RandomForestClassifier),

                   None,

                   ClassifierInput(4, 'SVC', "sklearn", SVC),

                   ClassifierInput(5, 'NB', "sklearn", GaussianNB),

                   ClassifierInput(6, 'LDA', "sklearn", LinearDiscriminantAnalysis),

                   ClassifierInput(7, 'CART', "sklearn", DecisionTreeClassifier),

                   ClassifierInput(8, 'KNN', "sklearn", KNeighborsClassifier),

                   ClassifierInput(9, 'ExtraTrees', "sklearn", ExtraTreesClassifier),

                   None,

                   NeuralNetClassifier(11, 'NN', 'keras', None)

                  ]



    ClassifierParameters = [

                              (1, {} ),

                              (2, {'n_estimators' : 1000, 'max_depth' : 4, 'bootstrap' : False, 'verbose' : 1, 'criterion': 'gini' }),

                              (3, {'min_split_loss' : 5, 'n_estimators' : 1000, 'subsample' : 0.5, 'reg_lambda' : 10 }),

                              (4, {} ),

                              (5, {} ),

                              (6, {} ),

                              (7, {} ),

                              (8, {} ),

                              (9, {'n_estimators' : 50, 'min_samples_split' : 2, 'max_depth' : 3 } ),

                              (10, {} ),

                              (11, {} ),

                           ]



    @staticmethod

    def get_classifier(no):

      classifierInput = Classification.Classifiers[no-1].allocate()

      paramz = Classification.ClassifierParameters[no-1][1]

      classifierInput.set_classification_params(paramz)

      return classifierInput



DataFiles = [{'positives': '../input/gauq-couples-aspects-REAL-7deg-20000-noa2b-cdata4.csv', 

             'negatives': '../input/gauq-couples-aspects-RANDOMIZED-7deg-20000-noa2b-cdata4.csv'},

            ]



dsConcat, dsPositives, dsRndNegatives, X, y, feature_names = DataReader.prepare_dataset(DataFiles[0])



i = 0 #removed while loop from main code for demonstration

classifierModel = Classification.get_classifier(Classification.ClassifiersToRun[i])

classifierModel.set_vectors(dsConcat, dsPositives, dsRndNegatives, X, y, feature_names)
DataValidation.plot_age_diff_distribution(dsConcat)

#the distributions should look identical
kf = StratifiedKFold(n_splits = Options.CrossValidationFoldCount, shuffle = False, random_state = Options.CrossValidationRandomState) 

cv_index = 0

cv_results = []

cv_accuracy = []





for train_index, test_index in kf.split(classifierModel.X, classifierModel.y):

  X_train_df, X_test_df = classifierModel.dfConcat.iloc[train_index, :], classifierModel.dfConcat.iloc[test_index, :]



  X_train, y_train, feature_names = DataReader.convert_to_array(X_train_df, True)

  X_test, y_test, feature_names_t = DataReader.convert_to_array(X_test_df, True)



  X_train, X_validate, y_train, y_validate = model_selection.train_test_split(X_train, y_train, test_size=

                                                                              (Options.ValidationDataSize / (Options.ValidationDataSize + Options.TrainDataSize)))



  X_train, X_validate, X_test = DataReader.transform_to_scores(X_train, X_validate, X_test, feature_names, 'float')



  summary_msg = ("\nSummary for cross validation fold " + str(cv_index + 1))

  DataValidation.print_data_summary(summary_msg, X_train, y_train, X_validate, y_validate, X_test, y_test)



  classifierModel.print_summary()

  fitted = classifierModel.fit(X_train, y_train, validation_data=(X_validate, y_validate))

  accuracy, y_pred = classifierModel.predict(X_test, y_test)



  if (cv_index == 0 or Options.CrossValidationShowAllFoldGraphs) :

    classifierModel.draw_confusion_matrix_data(y_test, y_pred, ['Random', 'Real'], 'Confusion matrix', normalize=False)

    classifierModel.draw_confusion_matrix_data(y_test, y_pred, ['Random', 'Real'], 'Confusion matrix (normalized)', normalize=True)

    classifierModel.draw_roc_curve(X_test, y_test)

    classifierModel.draw_epoch_loss(X_test, y_test)



  cv_accuracy.append(accuracy)



  cv_index = cv_index + 1



classifierModel.draw_cv_mean(cv_accuracy)

classifierModel.print_cv_details(cv_accuracy)