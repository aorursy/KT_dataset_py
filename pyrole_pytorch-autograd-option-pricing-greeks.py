import torch
x = torch.ones(2, 2, requires_grad=True)

print(x)
y = x + 2

print(y)
print(y.grad_fn)
z = y * y * 3

out = z.mean()



print(z, out)
a = torch.randn(2, 2)

a = ((a * 3) / (a - 1))

print(a.requires_grad)

a.requires_grad_(True)

print(a.requires_grad)

b = (a * a).sum()

print(b.grad_fn)
out.backward()

print(x.grad)
x = torch.randn(3, requires_grad=True)



y = x * 2

while y.data.norm() < 1000:

    y = y * 2



print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

y.backward(v)



print(x.grad)
print(x.requires_grad)

print((x ** 2).requires_grad)



with torch.no_grad():

	print((x ** 2).requires_grad)
print(x.requires_grad)

y = x.detach()

print(y.requires_grad)

print(x.eq(y).all())

import torch



x = torch.autograd.Variable(torch.Tensor([2]),requires_grad=True)

y = 5*x**4 + 3*x**3 + 7*x**2 + 9*x - 5



y.backward()

x.grad

## Plain Vanilla Call in TensorFlow

from scipy import stats

import numpy as np

def blackScholes_py(S_0, strike, time_to_expiry, implied_vol, riskfree_rate):

    S = S_0

    K = strike

    dt = time_to_expiry

    sigma = implied_vol

    r = riskfree_rate

    Phi = stats.norm.cdf

    d_1 = (np.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))

    d_2 = d_1 - sigma*np.sqrt(dt)

    return S*Phi(d_1) - K*np.exp(-r*dt)*Phi(d_2)
%%time

S_0 = 100

K = 101

T = 1

sigma = 0.3

r = 0.01



npv_numpy = blackScholes_py(S_0, K, T, sigma, r)
import torch

def blackScholes_pyTorch(S_0, strike, time_to_expiry, implied_vol, riskfree_rate):

    S = S_0

    K = strike

    dt = time_to_expiry

    sigma = implied_vol

    r = riskfree_rate

    Phi = torch.distributions.Normal(0,1).cdf

    d_1 = (torch.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*torch.sqrt(dt))

    d_2 = d_1 - sigma*torch.sqrt(dt)

    return S*Phi(d_1) - K*torch.exp(-r*dt)*Phi(d_2)
%%time

S_0 = torch.tensor([100.], requires_grad=True)

K = torch.tensor([101.], requires_grad=True)

T = torch.tensor([1.], requires_grad=True)

sigma = torch.tensor([0.3], requires_grad=True)

r = torch.tensor([0.01], requires_grad=True)



npv_pytorch = blackScholes_pyTorch(S_0, K, T, sigma, r)
%%time

S_0 = torch.tensor([100.],requires_grad=True)

K = torch.tensor([101.],requires_grad=True)

T = torch.tensor([1.],requires_grad=True)

sigma = torch.tensor([0.3],requires_grad=True)

r = torch.tensor([0.01],requires_grad=True)

npv_pytorch = blackScholes_pyTorch(S_0, K, T, sigma, r)

npv_pytorch.backward(retain_graph=True)
print(S_0.grad, K.grad, T.grad, r.grad, sigma.grad)
%%time

gradient = torch.autograd.grad(npv_pytorch, S_0, create_graph=True)

delta, =  gradient

delta.backward(retain_graph=True)

print('Delta: ', delta)

print('Gamma', S_0.grad)

def monte_carlo_down_out_py(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):

    stdnorm_random_variates = np.random.randn(samples, steps)

    S = S_0

    K = strike

    dt = time_to_expiry / stdnorm_random_variates.shape[1]

    sigma = implied_vol

    r = riskfree_rate

    B = barrier

    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet

    B_shift = B*np.exp(0.5826*sigma*np.sqrt(dt))

    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis=1)

    non_touch = (np.min(S_T, axis=1) > B_shift)*1

    call_payout = np.maximum(S_T[:,-1] - K, 0)

    npv = np.mean(non_touch * call_payout)

    return np.exp(-time_to_expiry*r)*npv
%%time

monte_carlo_down_out_py(100.,110.,2.,0.2,0.03,90.,1000,100000)
def monte_carlo_down_out_torch(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):

    stdnorm_random_variates = torch.distributions.Normal(0,1).sample((samples, steps))

    S = S_0

    K = strike

    dt = time_to_expiry / stdnorm_random_variates.shape[1]

    sigma = implied_vol

    r = riskfree_rate

    B = barrier

    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet

    B_shift = B*torch.exp(0.5826*sigma*torch.sqrt(dt))

    S_T = S * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*stdnorm_random_variates), dim=1)

    non_touch = torch.min(S_T, dim=1)[0] > B_shift

    call_payout = S_T[:,-1] - K

    call_payout[call_payout<0]=0

    npv = torch.mean(non_touch.type(torch.FloatTensor) * call_payout)

    return torch.exp(-time_to_expiry*r)*npv
%%time

S = torch.tensor([100.],requires_grad=True)

K = torch.tensor([110.],requires_grad=True)

T = torch.tensor([2.],requires_grad=True)

sigma = torch.tensor([0.2],requires_grad=True)

r = torch.tensor([0.03],requires_grad=True)

B = torch.tensor([90.],requires_grad=True)

npv_torch_mc = monte_carlo_down_out_torch(S, K, T, sigma, r, B, 1000, 100000)

npv_torch_mc.backward()

print(S.grad, T.grad, sigma.grad, r.grad)
def monte_carlo_down_out_torch_cuda(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):

    stdnorm_random_variates = torch.cuda.FloatTensor(steps, samples).normal_()

    S = S_0

    K = strike

    dt = time_to_expiry / stdnorm_random_variates.shape[1]

    sigma = implied_vol

    r = riskfree_rate

    B = barrier

    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet

    B_shift = B*torch.exp(0.5826*sigma*torch.sqrt(dt))

    S_T = S * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*stdnorm_random_variates), dim=1)

    non_touch = torch.min(S_T, dim=1)[0] > B_shift

    non_touch = non_touch.type(torch.cuda.FloatTensor)

    call_payout = S_T[:,-1] - K

    call_payout[call_payout<0]=0

    npv = torch.mean(non_touch * call_payout)

    return torch.exp(-time_to_expiry*r)*npv
%%time

S = torch.tensor([100.],requires_grad=True, device='cuda')

K = torch.tensor([110.],requires_grad=True, device='cuda')

T = torch.tensor([2.],requires_grad=True, device='cuda')

sigma = torch.tensor([0.2],requires_grad=True, device='cuda')

r = torch.tensor([0.03],requires_grad=True, device='cuda')

B = torch.tensor([90.],requires_grad=True, device='cuda')

npv_torch_mc = monte_carlo_down_out_torch_cuda(S, K, T, sigma, r, B, 1000, 100000)

npv_torch_mc.backward()

print(S.grad, T.grad, sigma.grad, r.grad)