from kaggle_secrets import UserSecretsClient

secret_label = "hello"

secret_value = UserSecretsClient().get_secret(secret_label)
print(secret_value)
!echo "${secret_value}"
!echo $secret_value
!echo "$secret_value"
!echo {secret_value}
!echo "{secret_value}"