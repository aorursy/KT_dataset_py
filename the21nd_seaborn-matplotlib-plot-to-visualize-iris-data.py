for i in range(0, 50):
    print(i)
# Set your own project id here
PROJECT_ID = 'kubernetes-in-action-258215'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("ahoj")
# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

# code cell

1+1
