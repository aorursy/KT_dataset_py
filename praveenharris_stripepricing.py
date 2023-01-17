# Import dependancies
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.html.widgets import *
from IPython.display import HTML, display
def stripe_base_pricing(x):
    return 0.029*x+0.30

t1 = np.arange(0, 501, 10)
plt.figure(figsize=(8, 6), dpi=80, facecolor='w')
plt.xlabel("Total Cost of Transaction (CAD $)")
plt.ylabel("Amount Stripe Charges (CAD $)")

plt.plot(t1, stripe_base_pricing(t1), 'k')
plt.grid()
t1 = np.arange(100, 401)
def revenue(rev):
    plt.figure(figsize=(5, 5), dpi=80, facecolor='w')
    plt.xlabel("Total Cost of Transaction (CAD $)")
    plt.ylabel("Cost (CAD $)")
        
    # cost of customer transaction
    plt.plot(t1, t1, 'g', label='total transaction_cost')
    # how much stripe is charging for the transaction
    plt.plot(t1, stripe_base_pricing(t1), 'b', label='stripe cost')    
    # how much seller makes from this transaction
    net_money = t1-stripe_base_pricing(t1)
    seller_rev = (rev/100)*net_money
    plt.plot(t1, seller_rev, 'r', label='seller revenue')
    # how much we are charging for that transaction
    our_rev = net_money - (rev/100)*net_money
    plt.plot(t1, our_rev, 'm', label='our revenue')
    plt.grid()
    
    ax = plt.gca()
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), ncol=1)
    
    vals = [['total cost', 'stripe charge', 'sellers revenue', 'our revenue']]
    val = 50
    for i, stripe_cost, s_rev, o_rev in zip(t1, stripe_base_pricing(t1), seller_rev, our_rev):
        if i % val == 0:
            temp = [i, stripe_cost, s_rev, o_rev]
            temp = ['%.2f' % item for item in temp]
            vals.append(temp)
    display(HTML(
        '<table><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in vals)
            )
     ))
    

# interact(revenue, rev=widgets.IntSlider(min=0,max=100, value=95));
revenue(95)
t2 = np.arange(5000, 30001)
def revenue_v2(rev):
    plt.figure(figsize=(8, 6), dpi=80, facecolor='w')
    plt.xlabel("Sellers monthly revenue (CAD $)")
    plt.ylabel("how much will be charged (CAD $)")
    
    # total amounts sold
    plt.plot(t2, t2, 'g', label='total revenue')
    # how much stripe will be charging
    transactions_processed = 90   # three transactionper day
    average_item_cost = t2 / transactions_processed
    stripe_charge_monthly = (average_item_cost * 0.029 + 0.30) * transactions_processed
    plt.plot(t2, stripe_charge_monthly, 'b', label='stripe cost')
    # net revenue
    net_revenue = t2 - stripe_charge_monthly
    plt.plot(t2, net_revenue, 'y', label='net revenue')
    # sellers revenue
    seller_revenue = net_revenue * rev / 100
    plt.plot(t2, seller_revenue, 'r', label='seller revenue')
    # our revenue
    our_revenue = net_revenue - seller_revenue
    plt.plot(t2, our_revenue, 'm', label='our revenue')
    plt.grid()
    
    ax = plt.gca()
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), ncol=1)
    
    vals = [['total_sales_monthly', 'stripe_charge', 'net_revenue', 'seller_revenue', 'our_revenue']]
    val = 5000
    for i, stripe_cost, n_rev, s_rev, o_rev in zip(t2, stripe_charge_monthly, net_revenue, seller_revenue, our_revenue):
        if i % val == 0:
            temp = [i, stripe_cost, n_rev, s_rev, o_rev]
            temp = ['%.2f' % item for item in temp]
            vals.append(temp)
    display(HTML(
        '<table><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in vals)
            )
     ))
    

# interact(revenue_v2, rev=widgets.IntSlider(min=0,max=100, value=95));
revenue_v2(95)




