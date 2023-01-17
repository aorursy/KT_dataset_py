import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N,T))
        F[:,0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n,t] = np.dot(F[:,t-1], (self.A[:,n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N,T))
        X[:,-1:] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                X[n,t] = np.sum(X[:,t+1] * self.A[n,:] * self.B[:, obs_seq[t+1]])

        return X

    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:,-1])

    def state_path(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:,-1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state,-1], reversed(path)

    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:,0] = self.pi * self.B[:,obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:,t-1] * self.A[:,n] * self.B[n, obs_seq[t]]
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)

        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield(last_state)
        for i in range(T-1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]

    def simulate(self, T):

        def draw_from(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0],:])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1],:])
            observations[t] = draw_from(self.B[states[t],:])
        return observations,states

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        # 观察序列的长度T
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(o_1,o_2,...,o_t,q_t = s_i | hmm)
            # Initialize alpha
            # 获得所有前向传播节点值 alpha_t(i)
            alpha = self._forward(observations)

            # beta_t(i) = P(o_t+1,o_t+2,...,o_T | q_t = s_i , hmm)
            # Initialize beta
            # 获得所有后向传播节点值 beta_t(i)
            beta = self._backward(observations)

            # 计算 xi_t(i,j) -> xi(i,j,t)
            xi = np.zeros((n_states, n_states, n_samples - 1))
            # 在每个时刻
            for t in range(n_samples - 1):
                # 计算P(O | hmm)
                denom = sum(alpha[:, -1])
                for i in range(n_states):
                    # numer[1,:] = 行向量，alpha[i,t]=实数，slef.A[i,:] = 行向量
                    # self.B[:,observations[t+1]].T = 行向量,beta[:,t+1].T = 行向量
                    numer = alpha[i, t] * self.A[i, :] * self.B[:, observations[t + 1]].T * beta[:, t + 1].T
                    xi[i, :, t] = numer / denom

                # 计算gamma_t(i) 就是对j进行求和
                gamma = np.sum(xi, axis=1)
                # need final gamma elements for new B
                prod = (alpha[:, n_samples - 1] * beta[:, n_samples - 1]).reshape((-1, 1))
                # 合并T时刻的节点
                gamma = np.hstack((gamma, prod / np.sum(prod)))
                # 列向量
                newpi = gamma[:, 0]
                newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
                newB = np.copy(self.B)

                # 观测状态数
                num_levels = self.B.shape[1]
                sumgamma = np.sum(gamma, axis=1)
                for lev in range(num_levels):
                    mask = observations == lev
                    newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

                if np.max(abs(self.pi - newpi)) < criterion and \
                                np.max(abs(self.A - newA)) < criterion and \
                                np.max(abs(self.B - newB)) < criterion:
                    done = 1
                self.A[:], self.B[:], self.pi[:] = newA, newB, newpi

# 状态转移概率
A = np.array([
    [  0,   1,   0,   0],
    [0.4,   0, 0.6,   0],
    [  0, 0.4,   0, 0.6],
    [  0,   0, 0.5, 0.5],
])
# 状态观测概率
B = np.array([
    [0.5,0.5],
    [0.3,0.7],
    [0.6,0.4],
    [0.8,0.2],
])
# 初始状态概率
pi = np.array([0.25,0.25,0.25,0.25])


# 使用我们设定的模型参数生成观测序列
h = HMM(A, B, pi)
observations_data, states_data = h.simulate(1000)


# 训练模型
guess = HMM(np.full((4,4),0.25),np.full((4,2),0.5),np.full((4),0.25))
guess.baum_welch_train(observations_data)
print(guess.A)
print(guess.B)
print(guess.pi)

# 预测并打印正确率
states_out = guess.state_path(observations_data)[1]
p = 0.0
for s in states_data:
    if next(states_out) == s: p += 1

print(p / len(states_data))

# 测试预测函数
states_out = h.state_path(observations_data)[1]
p = 0.0
for s in states_data:
    if next(states_out) == s: p += 1

print(p / len(states_data))

