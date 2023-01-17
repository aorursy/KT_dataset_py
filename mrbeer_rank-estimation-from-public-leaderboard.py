import numpy as np # linear algebra
accuracy = 0.997

naive_accuracy = 0.926

n_samples = 1000000

n_not_naive = int(1000000 * (1-naive_accuracy))



# Writing the total numbers correct answers

answers = np.zeros((n_samples))

answers[:int(n_samples * accuracy)] = 1



print(np.mean(answers))
n_monte_carlo = 50  # 50 for good normal behaiviour

public_percent = 0.01

n_public_samples = int(n_samples * public_percent)

n_team_montecarlo = 50



# conducting a monte carlo experiment

team_std = []

for i in range(n_monte_carlo):

    team_scores = []

    for j in range(n_team_montecarlo):

        # team answers with random errors in the non naive part

        team_answers = answers.copy()

        team_answers[-(n_not_naive):] = np.random.choice(team_answers[-(n_not_naive):], n_not_naive)

        # random seed is the same for all the teams

        np.random.seed(10*i)

        public_samples = np.random.choice(team_answers, n_public_samples)  # choosing public_percent of samples

        team_scores.append(np.mean(public_samples))

    team_std.append(np.std(team_scores))  # calculating standard deviation of the experiment

team_mean_std = np.mean(team_std)
# Using normal distribution property

print("The standard deviation of the score is {0:.5f} ".format(team_mean_std))

print("Total score has ~68% of being inside the {0:.5f} to {1:.5f} interval".format(

    (accuracy - team_mean_std), 

    (accuracy + team_mean_std)))

print("Total score has ~95% of being inside the {0:.5f} to {1:.5f} interval".format(

    (accuracy - 2*team_mean_std), 

    (accuracy + 2*team_mean_std)))