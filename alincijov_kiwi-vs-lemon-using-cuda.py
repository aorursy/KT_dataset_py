import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import cv2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
base_path = '../input/fruit-recognition/train/train/'
kiwis = list(os.walk(base_path + 'Kiwi'))[0][2]
lemons = list(os.walk(base_path + 'Lemon'))[0][2]

kiwis_lemons = (kiwis, list(os.walk(base_path + 'Kiwi'))[0][0]) + (lemons, list(os.walk(base_path + 'Lemon'))[0][0])

labels = np.array([[1, 0]] * len(kiwis) + [[0, 1]] * len(lemons))
features = []

for kiwi in kiwis_lemons[0]:
    features.append(cv2.resize(cv2.imread(kiwis_lemons[1] + '/' + kiwi, cv2.IMREAD_GRAYSCALE), (28, 28)))
    
    
for lemon in kiwis_lemons[2]:
    features.append(cv2.resize(cv2.imread(kiwis_lemons[3] + '/' + lemon, cv2.IMREAD_GRAYSCALE), (28, 28)))
    
features = np.array(features)
features = features.reshape(len(features), 28, 28)
# normalize
features = features / 255.0
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
forward_mod = SourceModule("""
__global__ void f_conv_before_act(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 5*5*6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 5);
        const int i2 = ((i /= 5    ) % 5);
        const int i3 = ((i /= 5    ) % 6);
        const int i4 = ((i /= 6    ) % 24);
        const int i5 = ((i /= 24    ) % 24);

        atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
    }
}

__global__ void f_conv_bias(float preact[6][24][24], float bias[6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        preact[i1][i2][i3] += bias[i1];
    }
}

__device__ float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

__global__ void activation_function(float *input, float *output, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        output[i] = sigmoid(input[i]);
    }
}

__global__ void f_soft_before_act(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 4*4*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 4);
        const int i2 = ((i /= 4    ) % 4);
        const int i3 = ((i /= 4    ) % 6);
        const int i4 = ((i /= 6    ) % 6);
        const int i5 = ((i /= 6    ) % 6);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
    }
}

__global__ void f_soft_bias(float preact[6][6][6], float bias[1])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        preact[i1][i2][i3] += bias[0];
    }
}

__global__ void f_final_before_act(float input[6][6][6], float preact[2], float weight[2][6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 2*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 2);
        const int i2 = ((i /= 2    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
    }
}

__global__ void f_final_bias(float preact[2], float bias[2])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 2;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        preact[i] += bias[i];
    }
}

""")
conv_before_act = forward_mod.get_function("f_conv_before_act")
conv_bias = forward_mod.get_function("f_conv_bias")
activation_function = forward_mod.get_function("activation_function")
soft_before_act = forward_mod.get_function("f_soft_before_act")
soft_bias = forward_mod.get_function("f_soft_bias")
final_before_act = forward_mod.get_function("f_final_before_act")
final_bias = forward_mod.get_function("f_final_bias")
backprop_mod = SourceModule("""
__device__ float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

__global__ void grad_final(float d_weight[2][6][6][6], float d_preact[2], float p_output[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 2*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 2);
        const int i2 = ((i /= 2    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
    }
}

const static float learning_rate = 1.0E-03f;

__global__ void bias_final(float bias[2], float d_preact[2])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 2;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        bias[i] += learning_rate * d_preact[i];
    }
}

__global__ void output_soft(float d_output[6][6][6], float n_weight[2][6][6][6], float nd_preact[2])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 2*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 2);
        const int i2 = ((i /= 2    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
    }
}

__global__ void before_act_soft(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        const float o = sigmoid(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void grad_soft(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 1*4*4*6*6*6;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 1);
        const int i2 = ((i /= 1    ) % 4);
        const int i3 = ((i /= 4    ) % 4);
        const int i4 = ((i /= 4    ) % 6);
        const int i5 = ((i /= 6    ) % 6);
        const int i6 = ((i /= 6    ) % 6);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
    }
}

__global__ void bias_soft(float bias[1], float d_preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        atomicAdd(&bias[0], learning_rate * d_preact[i1][i2][i3] / d);
    }
}

__global__ void output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 1*4*4*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 1);
        const int i2 = ((i /= 1    ) % 4);
        const int i3 = ((i /= 4    ) % 4);
        const int i4 = ((i /= 4    ) % 6);
        const int i5 = ((i /= 6    ) % 6);
        const int i6 = ((i /= 6    ) % 6);

        atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void before_act_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        const float o = sigmoid(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void grad_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*5*5*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 5);
        const int i3 = ((i /= 5    ) % 5);
        const int i4 = ((i /= 5    ) % 24);
        const int i5 = ((i /= 24    ) % 24);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
    }
}

__global__ void bias_conv(float bias[6], float d_preact[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        atomicAdd(&bias[i1], learning_rate * d_preact[i1][i2][i3] / d);
    }
}

__global__ void apply_gradient(float *output, float *grad, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        output[i] += learning_rate * grad[i];
    }
}

""")
grad_final = backprop_mod.get_function("grad_final")
bias_final = backprop_mod.get_function("bias_final")
output_soft = backprop_mod.get_function("output_soft")
before_act_soft = backprop_mod.get_function("before_act_soft")
grad_soft = backprop_mod.get_function("grad_soft")
bias_soft = backprop_mod.get_function("bias_soft")
output_conv = backprop_mod.get_function("output_conv")
before_act_conv = backprop_mod.get_function("before_act_conv")
grad_conv = backprop_mod.get_function("grad_conv")
bias_conv = backprop_mod.get_function("bias_conv")
apply_gradient = backprop_mod.get_function("apply_gradient")
error_mod = SourceModule("""
__global__ void calc_error(float *error, float *output, float *label, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        error[i] = label[i] - output[i];
    }
}
""")
calc_error = error_mod.get_function("calc_error")
np.random.seed(3243234242)

L1_preact = np.zeros((6, 24, 24)).astype(np.float32)
L1_weight = np.random.randn(6,5,5).astype(np.float32)
L1_bias = np.random.randn(6).astype(np.float32)
L1_O = np.int32(3456)

L1_opt = np.zeros((6, 24, 24)).astype(np.float32)
L2_preact = np.zeros((6, 6, 6)).astype(np.float32)
L2_weight = np.random.randn(1, 4, 4).astype(np.float32)
L2_bias = np.random.randn(1).astype(np.float32)
L2_opt = np.zeros((6, 6, 6)).astype(np.float32)
L2_O = np.int32(216)

L3_preact = np.zeros((2)).astype(np.float32)
L3_weight = np.random.randn(2,6,6,6).astype(np.float32)
L3_bias = np.zeros((2)).astype(np.float32)
L3_O = np.int32(2)

R = np.zeros((2)).astype(np.float32)
for epoch in tqdm(range(500)):
    
    for idx in range(len(X_train)):

        x = X_train[idx].astype(np.float32)
        y = y_train[idx].astype(np.float32)

        # Forward
        conv_before_act(cuda.In(x), cuda.InOut(L1_preact), cuda.In(L1_weight), block=(64,1,1), grid=(64,1))
        conv_bias(cuda.In(L1_preact), cuda.InOut(L1_bias), block=(64,1,1), grid=(64,1))
        activation_function(cuda.In(L1_preact), cuda.InOut(L1_opt), L1_O, block=(64,1,1), grid=(64,1))

        soft_before_act(cuda.In(L1_opt), cuda.InOut(L2_preact), cuda.In(L2_weight), block=(64,1,1), grid=(64,1))
        soft_bias(cuda.In(L2_preact), cuda.Out(L2_bias), block=(64,1,1), grid=(64,1))
        activation_function(cuda.In(L2_preact), cuda.InOut(L2_opt), L2_O, block=(64,1,1), grid=(64,1))

        final_before_act(cuda.In(L2_opt), cuda.InOut(L3_preact), cuda.In(L3_weight), block=(64,1,1), grid=(64,1))
        final_bias(cuda.In(L3_preact), cuda.InOut(L3_bias), block=(64,1,1), grid=(64,1))
        activation_function(cuda.In(L3_preact), cuda.InOut(R), L3_O, block=(64,1,1), grid=(64,1))

        L3_bp_preact = np.zeros((2,)).astype(np.float32)
        L3_bp_weight = np.zeros((2,6,6,6)).astype(np.float32)
        L2_bp_opt = np.zeros((6,6,6)).astype(np.float32)
        L2_bp_preact = np.zeros((6,6,6)).astype(np.float32)
        L2_bp_weight = np.zeros((1,4,4)).astype(np.float32)
        L1_bp_opt = np.zeros((6,24,24)).astype(np.float32)
        L1_bp_preact = np.zeros((6,24,24)).astype(np.float32)
        L1_bp_weight = np.zeros((6,5,5)).astype(np.float32)

        # Backward
        calc_error(cuda.Out(L3_bp_preact), cuda.In(R), cuda.In(y), np.int32(2), block=(1,1,1), grid=(2,1))

        grad_final(cuda.Out(L3_bp_weight), cuda.In(L3_bp_preact), cuda.In(L2_opt), block=(64,1,1), grid=(64,1))
        bias_final(cuda.InOut(L3_bias), cuda.In(L3_bp_preact), block=(64,1,1), grid=(64,1))

        output_soft(cuda.InOut(L2_bp_opt), cuda.In(L3_weight), cuda.In(L3_bp_preact), block=(64,1,1), grid=(64,1))
        before_act_soft(cuda.InOut(L2_bp_preact), cuda.In(L2_bp_opt), cuda.In(L2_preact), block=(64,1,1), grid=(64,1))
        grad_soft(cuda.InOut(L2_bp_weight), cuda.In(L2_bp_preact), cuda.In(L1_opt), block=(64,1,1), grid=(64,1))
        bias_soft(cuda.InOut(L2_bias), cuda.In(L2_bp_preact), block=(64,1,1), grid=(64,1))

        output_conv(cuda.InOut(L1_bp_opt), cuda.In(L2_weight), cuda.In(L2_bp_preact), block=(64,1,1), grid=(64,1))
        before_act_conv(cuda.InOut(L1_bp_preact), cuda.In(L1_bp_opt), cuda.In(L1_preact), block=(64,1,1), grid=(64,1))
        grad_conv(cuda.InOut(L1_bp_weight), cuda.In(L1_bp_preact), cuda.In(x), block=(64,1,1), grid=(64,1))
        bias_conv(cuda.InOut(L1_bias), cuda.In(L1_bp_preact), block=(64,1,1), grid=(64,1))

        apply_gradient(cuda.InOut(L3_weight), cuda.In(L3_bp_weight), np.int32(216 * 2), block=(64,1,1), grid=(64,1))
        apply_gradient(cuda.InOut(L2_weight), cuda.In(L2_bp_weight), np.int32(16 * 1), block=(64,1,1), grid=(64,1))
        apply_gradient(cuda.InOut(L1_weight), cuda.In(L1_bp_weight), np.int32(25 * 6), block=(64,1,1), grid=(64,1))