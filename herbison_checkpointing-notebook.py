# restore my state: (loads hello var)

import dill

dill.load_session('/kaggle/input/state.db')
# the first time I ran this, I set this var, but now it's loaded from previous state

#hello = "Hello World"

print(hello) # prints "Hello World"
# the first time I ran this, I used this to get the state.db fie that I saved in the attached dataset.

#import dill

#dill.dump_session('state.db')