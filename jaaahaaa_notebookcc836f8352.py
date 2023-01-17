#https://epiforecasts.io/covid/posts/national/norway/

r_norway = 1.1

r_sweden = 1.1



r_infected_norway = 0.01



#Assuming ten fold higher in sweden

r_infected_sweden = 0.10



# How much was r lowered?

a_infected = 200000*r_infected_norway

next_cycle_norway = a_infected * r_norway

r_with_mask = (next_cycle_norway-1)/a_infected

print("Updated R0 for norway with mask would be:")

print(r_with_mask)



a_infected_sweden = 200000 * r_infected_sweden

next_cycle_sweden = a_infected_sweden * r_sweden

next_cycle_sweden_mask = a_infected_sweden * r_with_mask

print("Number of less infected wearing mask in one cycle:")

print(next_cycle_sweden-next_cycle_sweden_mask)



print("Number of less infected in sweden wearing mask after five cycles:")

res_mask = a_infected_sweden

for i in range(5):

    res_mask = res_mask*r_with_mask

    

res = a_infected_sweden

for i in range(5):

    res = res*r_sweden

    

print(res-res_mask)