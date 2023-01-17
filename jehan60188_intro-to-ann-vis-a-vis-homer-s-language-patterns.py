import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import sys

from collections import Counter

from sklearn.cross_validation import train_test_split
# Helper function to evaluate the total loss on the dataset

def calculate_loss(X,y,model,reg_lambda):

    num_examples = len(X) # training set size

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions

    z1 = X.dot(W1) + b1

    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss

    corect_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss (optional)

    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1./num_examples * data_loss

    

    

# Helper function to predict an output (0 or 1)

def predict(model, x):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation

    z1 = x.dot(W1) + b1

    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

    

# This function learns parameters for the neural network and returns the model.

# - nn_hdim: Number of nodes in the hidden layer

# - num_passes: Number of passes through the training data for gradient descent

# - print_loss: If True, print the loss every 1000 iterations

# - epsilon: learning rate for gradient descent

def build_model(X,y,nn_hdim, epsilon,reg_lambda, num_passes=20000, print_loss=False):



    nn_input_dim = X.shape[1] # input layer dimensionality (number of datapoints per entry)

    nn_output_dim = len(np.unique(y)) # output layer dimensionality (number of classes)

    num_examples = len(X) # training set size

    # Initialize the parameters to random values. We need to learn these.

    np.random.seed(0)

    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)

    b1 = np.zeros((1, nn_hdim))

    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)

    b2 = np.zeros((1, nn_output_dim))

 

    # This is what we return at the end

    model = {}

     

    # Gradient descent. For each batch...

    for i in range(0, num_passes):

        

        # Forward propagation

        z1 = X.dot(W1) + b1

        a1 = np.tanh(z1)

        z2 = a1.dot(W2) + b2

        exp_scores = np.exp(z2)

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

 

        # Backpropagation

        delta3 = probs

        delta3[range(num_examples), y] -= 1

        dW2 = (a1.T).dot(delta3)

        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))

        dW1 = np.dot(X.T, delta2)

        db1 = np.sum(delta2, axis=0)

 

        # Add regularization terms (b1 and b2 don't have regularization terms)

        dW2 += reg_lambda * W2

        dW1 += reg_lambda * W1

 

        # Gradient descent parameter update

        W1 += -epsilon * dW1

        b1 += -epsilon * db1

        W2 += -epsilon * dW2

        b2 += -epsilon * db2

         

        # Assign new parameters to the model

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

         

        # Optionally print the loss.

        # This is expensive because it uses the whole dataset, so we don't want to do it too often.

        if print_loss and i % 1000 == 0:

          print( "Loss after iteration %i: %f" %(i, calculate_loss(X,y,model,reg_lambda)))

     

    return model



# Helper function to plot a decision boundary.

# If you don't fully understand this function don't worry, it just generates the contour plot below.

def plot_decision_boundary(X,y,pred_func):

    # Set min and max values and give it some padding

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5



    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    plt.show()

    

## Generate a dataset and plot it

#np.random.seed(0)

#X, y = sklearn.datasets.make_moons(200, noise=0.20)

#

#iris = datasets.load_iris()

#X = iris.data[0:150]

#y = iris.target[0:150]
def main(X,y,epsilon=.01,regLamda=.01,numLayers=10,num_passes=40000):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    

    #plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=plt.cm.Spectral) 

    

    # Gradient descent parameters (I picked these by hand)

    #epsilon=.01 # learning rate for gradient descent

    #reg_lambda = 0.01 # regularization strength

        # Build a model with a 3-dimensional hidden layer

    model = build_model(X_train,y_train,numLayers, epsilon,regLamda,print_loss=True,num_passes=num_passes)

     

    # Plot the decision boundary

    #plot_decision_boundary(X_train,y_test,lambda x: predict(model, x))

    #plt.title("Decision Boundary for hidden layer size 3")

    y_predict = predict(model, X_test)

    i=0

    wrongGues =0

    while i< len(y_test):

        if(y_predict[i] != y_test[i]):

            wrongGues = wrongGues +1

        i=i+1

    return [wrongGues/len(y_test),model]
def getLetterPairs(X1):

    allPairs = []

    helperVar1 =['a','b','c','d','e','f','g','h',

                 'i','j','k','l','m','n','o','p','q',

                 'r','s','t','u','v','w','x','y','z',

                 ' ','1','2','3','4','5','6','7','8','9','0']

    #get all possible letter pairs

    for x in helperVar1 :

        for y in helperVar1 :            

            allPairs.append(x+y)

#alternative technique

#    allPairs = []

#    for word in X1:

#        i=0

#        while(i<len(word)-1):

#            allPairs.append(word[i:i+2])

#            i=i+1

    allPairsU = np.unique(allPairs)

    

    wordPairCounts = []

    for word in X1:

        i=0

        letterPairCounter=[]

        while(i<len(word)-1):

            junk = np.where(allPairsU==word[i:i+2])

            if(len(junk[0])>0):

                letterPairCounter.append(junk[0][0])

            i=i+1

        wordPairCounts.append(letterPairCounter)  

    paddGoal = max([len(x) for x in wordPairCounts])+1

    j=0

    for word in wordPairCounts:

        i=len(word)

        while(len(word)<paddGoal):

            word.append(0)

            i=i+1

        word = np.array(word)

        wordPairCounts[j] = word

        j=j+1

    return [np.array(wordPairCounts),paddGoal,allPairsU]
def condition(word,allPairsU,padGoal):

    i=0

    letterPairCounter=[]

    while(i<len(word)-1):

        junk = np.where(allPairsU==word[i:i+2])

        letterPairCounter.append(junk[0][0])

        i=i+1

    i=len(letterPairCounter)

    while(len(letterPairCounter)<padGoal):

        letterPairCounter.append(0)

        i=i+1

    letterPairCounter = np.array(letterPairCounter)    

    return letterPairCounter
df = pd.read_csv("../input/simpsons_script_lines.csv",error_bad_lines=False,warn_bad_lines=False  )

homer = df[df["raw_character_text"] == "Homer Simpson"]

homer.shape[0]





homer = homer.sort('episode_id')

episodes = np.unique(homer['episode_id'].values)



homer.head()


#count words per episode

WC_array=[] 

Word_array = [] #also, let's just see what words he uses

for EP in episodes:

    homer_episode = homer[homer['episode_id']==EP]

    H_values = homer_episode['normalized_text'].values

    wordCount = 0

    for H_line in H_values:

        wordCount = wordCount + len(str(H_line).split())

        for s in str(H_line).split():

            Word_array.append(s)

    WC_array.append(wordCount) #words per episode



#make bag of worsds (for later)

Word_arrayU = np.unique(Word_array)

counter = Counter(Word_array)
#train NN

#get letter-pairs

X1=['get','gift','brave','pool','turn','grave','smash','voice','dorm','bar',

    'run','peak','flash','nail','spell','fish','braid','fleet','flex','fast',

    'drown','stun','pat','gear','hurt','fund','brand','hurl','shed','foot',

    'reach','crash','ward','flawed','pin','chaos','black','chip','breeze',

    'bird','scratch','root','snail','duck','shelf','claim','sick','truck',

    'lead','pop','drug','monk','lake','grain','pound','dark','lamp','beer',

    'choose','soil','horn','neck','dive','soup','flock','bland','stress',

    'disk','mud','deep','nerve','horse','name','flood','line','pour','hot',

    'swop','whip','weave','goat','trial','spot','chest','list','glare','slant',

    'key','like','club','count','orbit','inflate','tile','structure','normal',

    'inject','organ','guitar','pension','legend','freighter','engine','install',

    'break down','scramble','indulge','ready','bacon','tissue','feedback',

    'worry','union','relief','congress','football','number','winter','pocket',

    'research','danger','pepper','opposed','lazy','meaning','linear','castle',

    'earthquake','picture','flower','physics','regret','remind','training',

    'stadium','index','brainstorm','function','revoke','pastel','essay',

    'resort','movie','obese','cluster','mosaic','finance','matter','diagram',

    'manual','empire','broken','patient','rainbow','unfair','donor','panel',

    'contain','valid','release','council','decade','abbey','tourist',

    'reflect','biscuit','desert','single','joystick','divide','studio',

    'virtue','midnight','cereal','ritual','oppose','season','bowel','value',

    'kettle','flavor','gossip','ignite','lawyer','margin','ballet','exclude',

    'quarrel','perfect','liver','dismiss','report','buffet','dentist','follow',

    'enhance','banner','absorb','action','river','conflict','border','shoulder',

    'borrow','sugar','enter','glory','refund','rule','temple','orange',

    'describe','hover','purpose','pumpkin','verdict','upset','husband',

    'harbor','mercy','motif','depend','pigeon','mountain','outside',

    'privacy','retailer','unity','factory','nursery','cooperate',

    'particle','disaster','tendency','overview','abortion','multiply',

    'position','fabricate','delicate','expertise','adoption','minimize',

    'opera','hemisphere','projection','anxiety','equinox','dismissal',

    'socialist','formula','recommend','motorist','retiree','grateful',

    'exemption','agreement','abandon','calendar','photograph','genrate',

    'reduction','policy','courtesy','convenience','society'

    ,'equation','baseball','useful','pedestrian','multimedia',

    'negotiation','announcement','litigation','democratic','elaborate',

    'exploration','constellation','hostility','advertising','redundancy',

    'disagreement','deprivation','separation','satisfaction','registration',

    'excitement','epicalyx','available','empirical','distributor','conventional',

    'dictionary','intelligence','environment','lifestyle','security']



X2,padGoal,allPairsU = getLetterPairs(X1)

y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,

     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,

     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

     2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,

     3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

y = np.array(y)

y=y-1
#findTheBestModel

epsilon=.1

regLamda=.1

numLayers = 3

num_passes = 1000

M_rate = 1

rate=1

while(epsilon>.0001):

    if M_rate < .3:

        break

    while(regLamda>.0001):

        if M_rate < .3:

            break

        while(numLayers<100):

            if M_rate < .3:

                break

            while(num_passes<1000000):

                if M_rate < .3:

                    break

                rate,model= main(X2,y,epsilon,regLamda,numLayers,num_passes)

                print(M_rate)

                if rate < M_rate:

                    M_rate = rate

                    M_model = model

                    M_epsilon = epsilon

                    M_regLamda=regLamda

                    M_numLayers = numLayers

                    M_num_passes = num_passes

                epsilon = epsilon*.75

                regLamda = regLamda*.75

                numLayers = int(numLayers*1.5)

                num_passes = int(num_passes*1.2)
#count syllables per episode

SylArr=[] 

Word_array = [] #also, let's just see what words he uses

for EP in episodes:

    homer_episode = homer[homer['episode_id']==EP]

    H_values = homer_episode['normalized_text'].values

    sylablecount = 0

    for H_line in H_values:

        for word in str(H_line).split():

            letterPaircounter = condition(word,allPairsU,padGoal)

            syllables = predict(M_model,letterPaircounter)

            sylablecount = sylablecount + syllables

    SylArr.append(sylablecount)

    

#plot syllables/word vs episode

i=0

plotArr =[]

while i< len(WC_array):

    plotArr.append(float(SylArr[i][0])/float(WC_array[i]))

    i=i+1

plt.plot(plotArr)

plt.show()