'''

# Download command file

$ cd ~

$ wget -O gdrive https://drive.google.com/uc?id=0B3X9GlR6Embnb095MGxEYmJhY2c

$ chmod 755 ~/gdrive

$ mv ~/gdrive ~/.local/bin





# Setting

# Open url on your local machine, and input verification code



$ gdrive



# => Go to the following link in your browser:

# => https://accounts.google.com/o/oauth2/auth?client_id=xxx.apps.googleusercontent.com...





# Download files to Colfax Cluster



$ gdrive list | grep .pdf

# => Id                                                              Title                                      Size       Created

# => 0B2kJp7wSl9SIOUxvdnBxMmNIQXM                                    Intel_Deep_Learing_...ool_User_Guide.pdf   211.0 KB   2017-03-28 04:21:15



$ gdrive download -i 0B2kJp7wSl9SIOUxvdnBxMmNIQXM

Downloaded 'Intel_Deep_Learing_SDK_Deployment_Tool_User_Guide.pdf' at 211.0 KB/s, total 211.0 KB





# Upload files to Google gDrive

$ gdrive upload --file test.txt





# Extra: Encrypt files (on Colfax Cluster or your local machine)

# Check Google policies about file's license: https://www.google.com/policies/terms/

# Of course, you can use other encrypt ways, you don't have to encrypt public files.



# Encrypt

$ openssl aes-256-cbc -e -in input_file.txt -out encrypt_file.txt -pass pass:password0123



# Decypt

$ openssl aes-256-cbc -d -in encrypt_file.txt -out input_file.txt -pass pass:password0123

'''

pass