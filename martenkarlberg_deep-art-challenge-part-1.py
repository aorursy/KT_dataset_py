from dcgan import DCGAN

IMAGE_DIR = "../input/segmented-bob-ross-images/train/images"
hyperparameters = {
    "leaky_relu_slope": 0.0,
    "dropout_rate": 0.0,
    "weight_init_std": 0.5,
    "weight_init_mean": 0.0,
    "learning_rate_initial_discriminator": 0.0,
    "learning_rate_initial_generator": 0.0,
    "noise_array_dimensions": 50,
    "batch_size": 16,
    "label_smoothing": False,
    "label_noise": False
}
dcgan = DCGAN(hyperparameters)
dataset = dcgan.create_dataset(IMAGE_DIR)
generator = dcgan.build_generator()
discriminator = dcgan.build_discriminator()
dcgan.train(
    dataset,
    generator, 
    discriminator,
    epochs=5000,
    save_every_x_results=100)