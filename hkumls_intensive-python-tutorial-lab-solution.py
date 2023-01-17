# Welcome to the Intensive Python Tutorial Lab!
# 
# Just follow the instructions and comments...
# You must run every piece of code! Every part is important!
#
''' Code that looks like this are questions '''
''' You must try to fill in the blanks in these places '''
#
# Have fun!
#
# P.S. Run this piece of code too!
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



import matplotlib.pyplot as plt
import matplotlib.lines as mlines

magic_model, magic_line, magic_points = None, None, None
def magic_new_line(b, e):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    ymax = e + b * xmax
    ymin = e + b * xmin

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
def magic_tutorial_lab_plot():
    global magic_model, magic_line, magic_points
    if magic_model != target_model:
        plt.clf()
        training_xs = [ x for (x, y) in training_samples[:30] ]
        training_ys = [ y for (x, y) in training_samples[:30] ]
        magic_points = plt.scatter(training_xs, training_ys, label='30 Training samples')
    if magic_line != None:
        magic_line.remove()
    magic_line = magic_new_line(b, e)
    plt.legend()

from IPython.display import HTML
HTML('''
<script>
   document.querySelector('.input').style.display = 'none'
   document.querySelector('.output').style.display = 'none'
</script>
''')
# Boring stuff
# Run this code to import the packages!
import math
import random

print('Your packages have arrived')
# Run me several times!
# Guess what do I do?
print(random.uniform(-10, +10))
# Run me several times as well!
# Guess what do I do?
print(random.choice(['hins', 'henry', 'machine', 'learning']))
# Run me several times as well!
# Guess what do I do?
print(random.normalvariate(0, 10))
# Run me!
# Guess what do I do too?
print(math.sqrt(1))
print(math.sqrt(2))
print(math.sqrt(3))
print(math.sqrt(4))
def do_noise(signal, noise):
    return random.normalvariate(signal, noise)

print(do_noise(5,1))
print(do_noise(5,1))
print(do_noise(5,1))
print(do_noise(5,1))
print(do_noise(5,1))
b = 5
e = 100
def model(x):
    return b * x + e

print(model(0))
print(model(5))
print(model(10))
print(model(50))
print(model(100))
def linear_model(b, e):
    def model(x):
        return b * x + e
    return model

# Let's try our linear_model()!
# Make a model_a with b=x5, e=100
# Does it work?
model_a = linear_model(5, 100)
print(model_a(0))
print(model_a(5))
print(model_a(10))
print(model_a(50))
print(model_a(100))
def generate_samples(model, n, x_range, noise=0):
    x_min, x_max = x_range # Find out the min and max for the x...

    xs = [] # Make a list of all the xs
    for _ in range(n): # For n number of times...
        xs = xs + [ random.uniform(x_min, x_max) ] # Add a new random x to the list!
    ys = [ model(x) for x in xs ]

    # Zip together the xs and ys to make a list of all (x, y)s...
    # Then add noise to each x and y!
    return [ (do_noise(x, noise), do_noise(y, noise)) for (x, y) in zip(xs, ys) ]

# Let's generate a sample of size 100 from model_a!
test_samples = generate_samples(model_a, 100, (-100, +100), noise=5)
print(test_samples)
def mean_squared_error(samples, model):
    xs = [ x for (x, y) in samples ]
    ys = [ y for (x, y) in samples ]
    predicted_ys = [ model(x) for x in xs ]
    
    # Zip together the predicted_ys and ys to make a list of all (predicted_y, y)s...
    # Then take the difference and square!
    errors = [ (predicted_y - y)**2 for (predicted_y, y) in zip(predicted_ys, ys) ]
    return sum(errors) / len(errors)

# Remember, model_a is a linear_model(5, 100)
print(mean_squared_error(test_samples, model_a)) # Let's see how the actual model fits the sample
print(mean_squared_error(test_samples, linear_model(6, 100))) # Let's try with other models...
print(mean_squared_error(test_samples, linear_model(5, 300)))
def standard_error(samples, model):
    return math.sqrt(mean_squared_error(samples, model))
        
# Let's try our tests again!
print(standard_error(test_samples, model_a))
print(standard_error(test_samples, linear_model(6, 100)))
print(standard_error(test_samples, linear_model(5, 300)))
# Now that's it! Let's try fitting a linear model with backpropagation!

target_model = linear_model(50, 5)

training_samples = generate_samples(target_model, 1000, (-50, +50), noise=1)
validation_samples = generate_samples(target_model, 200, (-100, +100), noise=0)
    
b = seed_b = random.uniform(-100, 100)
e = seed_e = random.uniform(-100, 100)

for i in range(100):
    # b, e = update_linear_model(b, e, i)
    # Oops! We don't know how to update our linear model yet...
    
    print('Iteration', i + 1)
    print('Estimated beta:', b, '; Estimated epsilon:', e)
    print('Standard error:', standard_error(validation_samples, linear_model(b, e)))
    
print('Seed beta:', seed_b, '; Seed epsilon:', seed_e)
def update_linear_model(b, e, i):
    model = linear_model(b, e) # We find out the model...
    
    sample = random.choice(training_samples) # Grab a random training sample...
    x, y = sample # Grab its value...
    
    error = y - model(x) # And see how it differs with our model
    
    delta_b = error * x * b_learning_rate(i) # Backpropagation for weights!
    delta_e = error * e_learning_rate(i) # Backpropagation for biases!
    
    return b + delta_b, e + delta_e # Return new (b, e) for updating!

# One last thing...
# How fast should our network learn in each direction?
def b_learning_rate(iteration):
    return 0.00007 # At a constant rate?
def e_learning_rate(iteration):
    # Or a changing rate?
    if iteration < 40:
        return b_learning_rate(iteration)
    else:
        return 0.1 * min(1, (100 - iteration) / 20)
    
print('Updated your update_linear_model()!')
print('On iteration 1,', 'b learns at:', b_learning_rate(0), ', and e learns at:', e_learning_rate(0))
print('On iteration 5,', 'b learns at:', b_learning_rate(4), ', and e learns at:', e_learning_rate(4))
print('On iteration 10,', 'b learns at:', b_learning_rate(9), ', and e learns at:', e_learning_rate(9))
print('On iteration 50,', 'b learns at:', b_learning_rate(49), ', and e learns at:', e_learning_rate(49))
print('On iteration 100,', 'b learns at:', b_learning_rate(99), ', and e learns at:', e_learning_rate(99))
# Now let it run!
# Try to adjust the learning rates, update algorithms to see how different algorithms perform

target_model = linear_model(50, 5)

training_samples = generate_samples(target_model, 1000, (-50, +50), noise=1)
validation_samples = generate_samples(target_model, 200, (-100, +100), noise=0)
    
b = seed_b = random.uniform(-100, 100)
e = seed_e = random.uniform(-100, 100)

for i in range(100):
    b, e = update_linear_model(b, e, i)
    
    print('Iteration', i + 1)
    print('Estimated beta:', b, '; Estimated epsilon:', e)
    print('Standard error:', standard_error(validation_samples, linear_model(b, e)))
    
print('Seed beta:', seed_b, '; Seed epsilon:', seed_e)

magic_tutorial_lab_plot()