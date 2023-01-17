import torch
torch.__version__
import torch
import numpy as np
import torch.nn.functional as F
import random
import time

PATTERNSIZE = 20        # Size of the patterns to memorize 
NBNEUR = PATTERNSIZE    # One neuron per pattern element
NBPATTERNS = 5          # The number of patterns to learn in each episode
NBPRESCYCLES = 2        # Number of times each pattern is to be presented
PRESTIME = 6            # Number of time steps for each presentation
PRESTIMETEST = 6        # Same thing but for the final test pattern
INTERPRESDELAY = 4      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode

# Generate the full list of inputs, as well as the target output at last time step, for an episode. 
def generateInputsAndTarget():
    inputT = np.zeros((NBSTEPS, 1, NBNEUR)) #inputTensor, initially in numpy format
    # Create the random patterns to be memorized in an episode
    patterns=[]
    for nump in range(NBPATTERNS):
        patterns.append(2*np.random.randint(2, size=PATTERNSIZE)-1)
    # Building the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    degradedtestpattern = testpattern * np.random.randint(2, size=PATTERNSIZE)
    # Inserting the inputs in the input tensor at the proper places
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi =nc * (NBPATTERNS * (PRESTIME+INTERPRESDELAY)) + ii * (PRESTIME+INTERPRESDELAY) + nn
                inputT[numi][0][:] = patterns[ii][:]
    # Inserting the degraded pattern
    for nn in range(PRESTIMETEST):
        inputT[-PRESTIMETEST + nn][0][:] = degradedtestpattern[:]
    inputT = 20.0 * torch.from_numpy(inputT.astype(np.float32))  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern.astype(np.float32))
    return inputT, target
# Note that each column of w and alpha defines the inputs to a single neuron
w = (.01 * torch.randn(NBNEUR, NBNEUR)).requires_grad_() # Fixed weights
alpha = (.01 * torch.randn(NBNEUR, NBNEUR)).requires_grad_() # Plasticity coeffs.
optimizer = torch.optim.Adam([w, alpha], lr=3e-4)

total_loss = 0.0; all_losses = []
nowtime = time.time()
print("Starting episodes...")
store_loss = []
for numiter in range(3000): # Loop over episodes
    y = torch.zeros(1, NBNEUR, requires_grad=True) # Initialize neuron activations
    hebb = torch.zeros(NBNEUR, NBNEUR, requires_grad=True) # Initialize Hebbian trace
    inputs, target = generateInputsAndTarget() # Generate inputs & target for this episode
    optimizer.zero_grad()
    # Run the episode:
    for numstep in range(NBSTEPS):
        yout = torch.tanh( y.mm(w + torch.mul(alpha, hebb)) + inputs[numstep] )
        hebb = .99 * hebb + .01 * torch.ger(y[0], yout[0]) # torch.ger = Outer product
        y = yout
    # Episode done, now compute loss, apply backpropagation
    loss = (y[0] - target.requires_grad_()).pow(2).sum()
    loss.backward()
    optimizer.step()
    
    # Print statistics
    print_every = 10
    to = target.detach().numpy(); yo = y.data.detach().numpy()[0][:]
    z = (np.sign(yo) != np.sign(to)); lossnum = np.mean(z)  # Compute error rate
    total_loss  += lossnum
    if (numiter+1) % print_every == 0:
        previoustime = nowtime;  nowtime = time.time()
        print("Episode", numiter, "=== Time spent on last", print_every, "iters: ", nowtime - previoustime)
        print(target.detach().numpy()[-10:])   # Target pattern to be reconstructed
        print(inputs.detach().numpy()[numstep][0][-10:])  # Last input (degraded pattern)
        print(y.data.detach().numpy()[0][-10:])   # Final output of the network
        total_loss /= print_every
        print("Mean error rate over last", print_every, "iters:", total_loss, "\n")
        store_loss.append(total_loss)
        total_loss = 0
import matplotlib.pyplot as plt

plt.plot(store_loss, label="Loss")
plt.plot()
