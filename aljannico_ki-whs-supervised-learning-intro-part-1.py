# Trainingsdaten in der Form (x1,x2,y)

print("The OR problem:")

D = [(1,1,1),(1,0,1),(0,1,1),(0,0,-1)]

print("Data (last element is the class):",D)

def add_bias(X):

    res = []

    for d in X:

        res.append(tuple([1]+list(d))) # add 1 to the front of each tuple

    return res

            

D = add_bias(D)

print("Augmented Data:",D)



# 3 Weights

W = [-0.15, 0.2, 0.1] # Initialisiert wie oben

eta = 0.2

print("Weights:",W,", Eta:",eta)



def dot(a,b):

    return sum([i*j for i,j in zip(a,b)])



def compute(x):    

    return dot(x,W)



def classify(v):

    return 1 if v >= 0 else -1



print("\nPerceptron output for data:")

for d in D:

    print("  ",d," -> ", compute(d[0:-1]), "=>", classify(compute(d[0:-1])))  

    

# Implementing the learning Delta w_j =  eta * (y - y_hat) x_j

def delta(y,y_hat,x_j):

    return eta * (y-y_hat) * x_j

        

def check_samples():

    # Determine classification results for all elements of D (i.e. compute y_hat)

    classified = [list(d) + [classify(compute(d[0:-1]))] for d in D]

    correct = True

    for d in classified:

        # y_hat != y, an ERROR?

        if d[-1] != d[-2]:

            correct = False

            continue

    return correct,classified

    

MAX_ROUNDS = 10 # Don't try to converge after MAX_ROUNDS tries



''' The following algo stops as soon as all samples are correctly 

    classified (everything else makes no sense here, as we need

    an error to change weights!).

    Furthermore, we simply redo the classification as soon as the

    the weights have been changed (the continue below) and

    we restart to run over the samples in a fixed sequence,

    sample 0, sample 1, etc. each time. Thus, it may happen

    - if we do not converge - that we change and re-change

    the weight seeing always only sample 1 and 2 and never

    using an error on sample 3.

'''    

def learn():

    print("\nLearning new weights:")

    round = 0

    run = True

    while run and round < MAX_ROUNDS:

        check,classified = check_samples()

        if check:

            run = False

        else:

            # Update the weights if necessary

            for d in classified:

                # Note: d[-1] is y_hat, d[-2] is the original class y

                if d[-1] != d[-2]: # Element has wrong class 

                    print("  Wrong class: ",d, " -> adapt some weights")

                    for i in range(len(W)):

                        W[i] += delta(d[-2],d[-1],d[i])

                    print("  Weights: ",W)

                    continue # Do this only for the first erroneous sample

        round += 1

    return not run



if learn(): 

    print("\nWe successfully learned!")

else:

    print("\nWe still make errors after",MAX_ROUNDS,"rounds!")

    

print("Learned Weights: ",W)



# Apply this to an XOR-Problem

print("\n\nThe XOR problem:")

D = [(1,1,-1),(1,0,1),(0,1,1),(0,0,-1)]

print("Data (last element is the class):",D)

D = add_bias(D)

print("Augmented Data:",D)



# Simply use the old weights

if learn(): 

    print("\nWe successfully learned!")

else:

    print("\nWe still make errors after",MAX_ROUNDS,"rounds!")

    

print("Learned Weights: ",W)

            
# Trainingsdaten in der Form (x1,x2,y)

D = [(1,1,1),(1,-1,1),(-1,1,1),(-1,-1,-1)] # slightly modified OR

W = [0,0] # weights are now 2-dimensional 



def errors(weights):

    global W

    W = weights

    # Determine classification results for all elements of D (i.e. compute y_hat)

    classified = [list(d) + [classify(compute(d[0:-1]))] for d in D]

    error_count = 0

    for d in classified:

        # y_hat != y, an ERROR?

        if d[-1] != d[-2]:

            error_count += 1            

    return error_count



# Let us span a grid:

import numpy as np

m1_dim = np.linspace(-1,1,51)

m2_dim = np.linspace(-1,1,51)

result = [errors((x,y)) for x in m1_dim for y in m2_dim]

print(result)
from IPython.core.display import display, HTML

import json



def plot3D(X, Y, Z, height=600, xlabel = "X", ylabel = "Y", zlabel = "Z", initialCamera = None):



    options = {

        "width": "100%",

        "style": "surface",

        "showPerspective": True,

        "showGrid": True,

        "showShadow": False,

        "keepAspectRatio": True,

        "height": str(height) + "px",

        "xLabel": xlabel,

        "yLabel": ylabel,

        "zLabel": zlabel

    }



    if initialCamera:

        options["cameraPosition"] = initialCamera

      

    pos = 0

    data = []

    for x in X:

        for y in Y:

            if pos < 10: print(x,y,Z[pos])

            data.append({"x": x, "y": y, "z": Z[pos]})

            pos += 1

    

    print(data[0:10])

    # data = [ {"x": X[y,x], "y": Y[y,x], "z": Z[y,x]} for y in range(X.shape[0]) for x in range(X.shape[1]) ]

    visCode = r"""

       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />

       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>

       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>

       <div id="visualization"></div>

       <script type="text/javascript">

        var data = new vis.DataSet();

        data.add(""" + json.dumps(data) + """);

        var options = """ + json.dumps(options) + """;

        var container = document.getElementById("visualization");

        var graph3d = new vis.Graph3d(container, data, options);

        graph3d.on("cameraPositionChange", function(evt)

        {

            elem = document.getElementById("pos");

            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;

        });

       </script>

    """

    htmlCode = "<iframe srcdoc='"+visCode+"' width='100%' height='" + str(height) + "px' style='border:0;' scrolling='no'> </iframe>"

    display(HTML(htmlCode))



plot3D(m1_dim, m2_dim, result, height=500, xlabel="m_1", ylabel="m_2", zlabel="err") 

# kann bei zu großen Datensätzen zu einem Erreichen des DataLimits führen (seltsam)

# kann man aber erhöhen (wie genau?)

# Übrigens haben H,V und D mit der Kameraposition zu tun, kann man auch weglassen, klar.
# All little bit more precise. change to True if you want to see it, you may even try 501...

if False:

    m1_dim = np.linspace(-1,1,201)

    m2_dim = np.linspace(-1,1,201)

    result = [errors((x,y)) for x in m1_dim for y in m2_dim]

    plot3D(m1_dim, m2_dim, result, height=500, xlabel="m_1", ylabel="m_2", zlabel="err") 
def surrogate_errors(weights):

    global W

    W = weights

    result = 0

    for d in D:

        result += max(-d[-1]*dot(weights,d[0:-1]),0)                

    return result



m1_dim = np.linspace(-1,1,51)

m2_dim = np.linspace(-1,1,51)

result = [surrogate_errors((x,y)) for x in m1_dim for y in m2_dim]

# print(result)

plot3D(m1_dim, m2_dim, result, height=500, xlabel="m_1", ylabel="m_2", zlabel="err") 