import torch

# refer to https://stackoverflow.com/questions/54316053/update-step-in-pytorch-implementation-of-newtons-method

# inital x

initial_x = torch.tensor([4.], requires_grad = True) 



# function to want to solve

def solve_func(x): 

    return torch.exp(x) - 2



def newton_method(function, initial, iteration=10, convergence=0.0001):

    for i in range(iteration): 

        previous_data = initial.clone()

        value = function(initial)

        value.backward()

        # update 

        initial.data -= (value / initial.grad).data

        # zero out current gradient to hold new gradients in next iteration 

        initial.grad.data.zero_() 

        print("epoch {}, obtain {}".format(i, initial))

        # Check convergence. 

        # When difference current epoch result and previous one is less than 

        # convergence factor, return result.

        if torch.abs(initial - previous_data) < torch.tensor(convergence):

            print("break")

            return initial.data

    return initial.data # return our final after iteration



# call starts

result = newton_method(solve_func, initial_x)
print(result)