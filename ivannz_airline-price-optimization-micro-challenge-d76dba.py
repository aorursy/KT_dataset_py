import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

class BasePricePolicy(object):
    def __init__(self):
        pass

    def __call__(self, days_left, tickets_left, demand_level):
        # raise NotImplementedError()
        return demand_level - 10
class SimplePricePolicy(BasePricePolicy):
    def __init__(self):
        super(SimplePricePolicy, self).__init__()
    
    def __call__(self, days_left, tickets_left, demand_level):
        if days_left > 1:
            qty = 10
        else:
            qty = min(tickets_left, demand_level / 2)

        price = demand_level - qty
        return price
import numpy as np
from scipy.optimize import brentq

def opt_value(q_bar, A):
    if q_bar <= 0:
        return np.zeros_like(A[:, :1])

    # for the y-vector
    y = 0.5 * A / q_bar
    if y.ndim < 2:
        y = y.reshape(1, -1)

    # solve the opt problem
    gfun = lambda C, z: 1 - np.sum(np.maximum(z - C, 0))
    C_opt = np.array([[brentq(gfun, 0, max(max(x), 0), args=(x,))
                       if gfun(0, x) < 0 else 0 for x in y]]).T

    # get the solution and the value
    x_opt = np.maximum(y - C_opt, 0)
    V_opt = C_opt + 0.5 * np.sum(x_opt**2, axis=-1, keepdims=True)

    return 2 * q_bar * q_bar * V_opt
from scipy.optimize import fmin_cobyla

class SimulatedPricePolicy(BasePricePolicy):
    def __init__(self, n_simulations=100):
        super(SimulatedPricePolicy, self).__init__()
        self.n_simulations = n_simulations
    
    def __call__(self, days_left, tickets_left, demand_level):
        if days_left < 2:
            qty = min(tickets_left, demand_level / 2)
        else:
            A_future = np.random.uniform(100, 200, size=(self.n_simulations, days_left - 1))

            f_fun = lambda x: - ((demand_level - x) * x + opt_value(tickets_left - x, A_future).mean())
            gfun1 = lambda x: x
            gfun2 = lambda x: tickets_left - x
            qty_opt = fmin_cobyla(f_fun, tickets_left / 2, (gfun1, gfun2))

            qty = qty_opt.item()

        # end if

        price = demand_level - qty
        return price
class ApproxPricePolicy(BasePricePolicy):
    def __init__(self, demand_bins):
        super(ApproxPricePolicy, self).__init__()
        self.demand_bins = demand_bins

    def __call__(self, days_left, tickets_left, demand_level):
        tickets_left, days_left = int(tickets_left), int(days_left)
        self.compute_dp(days_left, tickets_left)

        # get the bin of the current demand level
        demand_level_bin = np.digitize(demand_level, self.demand_bins) - 1

        # the optimal quantity is just the argmax (by construction)
        qty = np.argmax(self.current_[demand_level_bin, :tickets_left + 1]
                        + self.value_[days_left - 1, tickets_left::-1])

        price = demand_level - qty
        return price

    def compute_dp(self, days_left, tickets_left):
        dp_computed_, tickets_left = hasattr(self, "value_"), int(tickets_left)
        if dp_computed_:
            n_days, n_tickets_p1 = self.value_.shape
            dp_computed_ = (n_days >= days_left) and (n_tickets_p1 > tickets_left)

        if dp_computed_:
            return

        # It is necessary to recompute
        self.value_, self.current_ = self._compute_dp(days_left, tickets_left)
    
    def _compute_dp(self, n_days, n_tickets):
        # compute (A - q) * q, q=0..n_tickets
        current = np.zeros((len(self.demand_bins), 1 + n_tickets), dtype=float)
        for q in range(1 + n_tickets):
            current[:, q] = (self.demand_bins - q) * q

        # Compute \mathbb{E}_A \max_{q\in [0, x]} V_{t+1}(x, q; A)
        #  for all x=0..n_tickets, t=1..n_days
        V_tilde = np.zeros((n_days, 1 + n_tickets), dtype=float)
        for t in range(1, n_days):
            # V_t(x, q; A) = (A - q) * q + \tilde{V}_{t+1}(x - q), q=0..x
            # V_t(x; A) = \max_{q=0}^x V_t(x, q, A)
            # \tilde{V}_t(x) = \mathbb{E}_A V_t(x; A)
            for x in range(1 + n_tickets):
                V_txq = current[:, :x + 1] + V_tilde[t - 1, np.newaxis, x::-1]
                V_tilde[t, x] = np.mean(np.max(V_txq, axis=-1), axis=0)
            # end for
        # end for
        return V_tilde, current
# pricing_function = SimplePricePolicy()
# pricing_function = SimulatedPricePolicy(100)
pricing_function = ApproxPricePolicy(np.linspace(100, 200, num=2001))
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)
