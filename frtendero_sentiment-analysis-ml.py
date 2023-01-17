# Read files

negativeReviewsFileName = '../input/rt-polarity.neg'

with open(negativeReviewsFileName, encoding='utf-8', errors='ignore') as f: 

    negativeReviews = f.readlines()



positiveReviewsFileName = '../input/rt-polarity.pos'

with open(positiveReviewsFileName, encoding='utf-8', errors='ignore') as f: 

    positiveReviews = f.readlines()



# Split Corpus into Training and Test Data

testTrainingSplitIndex = 2500

testNegativeReviews = negativeReviews[testTrainingSplitIndex+1:]

testPositiveReviews = positiveReviews[testTrainingSplitIndex+1:]

trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]

trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]
def getVocabulary():

    positiveWordList = [word for line in trainingPositiveReviews for word in line.split()]

    negativeWordList = [word for line in trainingNegativeReviews for word in line.split()]

    allWordList = [item for sublist in [positiveWordList, negativeWordList] for item in sublist]

    vocabulary = list(set(allWordList))

    return vocabulary
# use the function

vocabulary = getVocabulary()

vocabulary[0]
def extract_features(review):

    review_words = set(review)

    features = {}

    for word in vocabulary:

        features[word] = (word in review_words)

    return features
def getTrainingData():

    negTaggedTrainingReviewList = [

        {'review': oneReview.split(), 'label': 'negative'} for oneReview in trainingNegativeReviews]

    posTaggedTrainingReviewList = [

        {'review': oneReview.split(), 'label': 'positive'} for oneReview in trainingPositiveReviews]

    fullTaggedTrainingData = [

        item for sublist in [negTaggedTrainingReviewList, posTaggedTrainingReviewList] for item in sublist]

    trainingData = [(review['review'], review['label']) for review in fullTaggedTrainingData]

    return trainingData
trainingData = getTrainingData()

trainingData[0]
import nltk
def getTrainedNaiveBayesClassifier(extract_features, trainingData):

    trainingFeatures = nltk.classify.apply_features(extract_features, trainingData)

    trainedNBClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    return trainedNBClassifier
trainedNBClassifier = getTrainedNaiveBayesClassifier(extract_features, trainingData)
def naiveBayesSentimentCalculator(review):

    problemInstance = review.split()

    problemFeatures = extract_features(problemInstance)

    return trainedNBClassifier.classify(problemFeatures)
# test the function

naiveBayesSentimentCalculator("What an awesome movie")
naiveBayesSentimentCalculator("What a terrible movie")
def getTestReviewSentiments(naiveBayesSentimentCalculator):

    testNegResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]

    testPosResults = [naiveBayesSentimentCalculator(review) for review in testPositiveReviews]

    labelToNum = {'positive': 1, 'negative': -1}

    numericNegResults = [labelToNum[x] for x in testNegResults]

    numericPosResults = [labelToNum[x] for x in testPosResults]

    return {'results-on-positive': numericPosResults, 'results-on-negative': numericNegResults}
def runDiagnostics(reviewResult):

    positiveReviewsResult = reviewResult['results-on-positive']

    negativeReviewsResult = reviewResult['results-on-negative']

    numTruePositive = sum(x > 0 for x in positiveReviewsResult)

    numTrueNegative = sum(x < 0 for x in negativeReviewsResult)

    pctTruePositive = float(numTruePositive)/len(positiveReviewsResult)

    pctTrueNegative = float(numTrueNegative)/len(negativeReviewsResult)

    totalAccurate = numTruePositive + numTrueNegative

    total = len(positiveReviewsResult) + len(negativeReviewsResult)

    print("Accuracy on positive reviews = " + "%.2f" % (pctTruePositive*100) + "%")

    print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative*100) + "%")

    print("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")
# Obtain results

runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))